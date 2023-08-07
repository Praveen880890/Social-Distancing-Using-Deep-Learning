import collections
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
from IPython import display
from openvino import runtime as ov
from openvino.tools.mo.front import tf as ov_tf_front
from openvino.tools import mo

# sys.path.append("../utils")
import notebook_utils as utils
# A directory where the model will be downloaded.
base_model_dir = Path("model")

# The name of the model from Open Model Zoo
model_name = "ssdlite_mobilenet_v2"
## Uncomment the below section if you want to re download and convert the model into openvino , However it is done so this step is not neccessary to run.
# **** 
# archive_name = Path(f"{model_name}_coco_2018_05_09.tar.gz")
# model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/{model_name}/{archive_name}"

# # Download the archive
# downloaded_model_path = base_model_dir / archive_name
# if not downloaded_model_path.exists():
#     utils.download_file(model_url, downloaded_model_path.name, downloaded_model_path.parent)

# # Unpack the model
# tf_model_path = base_model_dir / archive_name.with_suffix("").stem / "frozen_inference_graph.pb"
# if not tf_model_path.exists():
#     with tarfile.open(downloaded_model_path) as file:
#         file.extractall(base_model_dir)
# ***


precision = "FP32"
# The output path for the conversion.
converted_model_path = Path("../models") / f"{model_name}_{precision.lower()}.xml"

# Convert it to IR if not previously converted
trans_config_path = Path(ov_tf_front.__file__).parent / "ssd_v2_support.json"
try:
    if not converted_model_path.exists():
        ov_model = mo.convert_model(
            tf_model_path, 
            compress_to_fp16=(precision == 'FP16'), 
            transformations_config=trans_config_path,
            tensorflow_object_detection_api_pipeline_config=tf_model_path.parent / "pipeline.config", 
            reverse_input_channels=True
        )
        ov.serialize(ov_model, converted_model_path)
        del ov_model
except:
    print("model is already created")
# Initialize OpenVINO Runtime.
ie_core = ov.Core()
# Read the network and corresponding weights from a file.
model = ie_core.read_model(model=converted_model_path)
# Compile the model for CPU (you can choose manually CPU, GPU etc.)
# or let the engine choose the best available device (AUTO).
compiled_model = ie_core.compile_model(model=model, device_name="CPU")

# Get the input and output nodes.
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Get the input size.
height, width = list(input_layer.shape)[1:3]

classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "hair brush"
]
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
# Colors for the classes above (Rainbow Color Map).
def dist(pt1,pt2):
    try:
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
    except:
        return

colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()


def process_results(frame, results, thresh=0.5):
    # The size of the original frame.
    h, w = frame.shape[:2]
    # The 'results' variable is a [1, 1, 100, 7] tensor.
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
      if int(label)==1:
        boxes.append(
            tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
        )
        labels.append(int(label))
        scores.append(float(score))

    # Apply non-maximum suppression to get rid of many overlapping entities.
    # See https://paperswithcode.com/method/non-maximum-suppression
    # This algorithm returns indices of objects to keep.
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.2
    )

    # If there are no boxes.
    if len(indices) == 0:
        return []

    # Filter detected objects.
    # return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]
    return indices,boxes


def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        # Choose color for the label.
        color = tuple(map(int, colors[label]))
        # Draw a box.
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=4)

        # Draw a label name inside the box.
        cv2.putText(
            img=frame,
            text=f"{classes[label]} {score:.2f}",
            org=(box[0] + 10, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1000,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame
# Main processing function to run object detection.
def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0):
    player = None
    try:
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(
            source=source, flip=flip, fps=80, skip_first_frames=0
        )
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('../demo_videos/output.mp4', fourcc, 20.0, (1280,720))
        processing_times = collections.deque()
        distance_thres = 100
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            # Resize the image and change dims to fit neural network input.
            input_img = cv2.resize(
                src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA
            )
            # Create a batch of images (size = 1).
            input_img = input_img[np.newaxis, ...]

            # Measure processing time.

            start_time = time.time()
            # Get the results.
            results = compiled_model([input_img])[output_layer]
            stop_time = time.time()
            # Get poses from network results.
            
            indexes,boxes = process_results(frame=frame, results=results)
            # Draw boxes on a frame.
            persons = []
            person_centres = []
            violate = set()
            label="person"
            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    persons.append(boxes[i])
                    person_centres.append([x+w//2,y+h//2])
    
            for i in range(len(persons)):
                for j in range(i+1,len(persons)):
                    if dist(person_centres[i],person_centres[j]) <= distance_thres:
                        violate.add(tuple(persons[i]))
                        violate.add(tuple(persons[j]))
    
            v = 0
            
            for (x,y,w,h) in persons:
                if (x,y,w,h) in violate:
                    color = (0,0,225)
                    v+=1
                else:
                    
                    color = (125,255,0)
    
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                # cv2.circle(frame,(x+w//2,y+h//2),2,(0,0,255),2)
                tl = 2 or round(0.002 * (frame.shape[0] + source_image.shape[1]) / 2) + 1  # line/font thickness
                color = color or [random.randint(0, 255) for _ in range(3)]
                cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),color,thickness=tl, lineType=cv2.LINE_AA)
                c1,c2=(int(x),int(y)),(int(x+w),int(y+h))
                if label=="person":
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            cv2.putText(frame,'Number of Violations : '+str(v),(20,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            # frame = draw_boxes(frame=frame, boxes=boxes)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(winname=title, mat=frame)
                out.write(frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)
            if key==ord("q"):
                
                break
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            out.release()
            cv2.destroyAllWindows()
video_file = "../data/pedestrians.mp4"

run_object_detection(source=video_file, flip=False, use_popup=True)