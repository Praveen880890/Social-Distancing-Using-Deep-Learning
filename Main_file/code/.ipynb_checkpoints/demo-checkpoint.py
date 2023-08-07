from openvino.runtime import Core
core = Core()
# Load the pre optimized model
yolov8n_with_preprocess_model = core.read_model('../models/yolov8n_openvino_int8_model/yolov8n_with_preprocess.xml',)


import json
# Load the label map
with open('../models/yolov8n_labels.json', 'r') as f:
    label_map = json.load(f)
import collections
from IPython import display
import cv2
import numpy as np
import time
from typing import Dict, Tuple
from ultralytics.yolo.utils.plotting import colors
from utils import VideoPlayer, detect_without_preprocess

distance_thres=100

def draw_results_new(results:Dict, source_image:np.ndarray, label_map:Dict):
    """
    Helper function for drawing bounding boxes on image
    Parameters:
        image_res (np.ndarray): detection predictions in format [x1, y1, x2, y2, score, label_id]
        source_image (np.ndarray): input image for drawing
        label_map; (Dict[int, str]): label_id to class name mapping
    Returns:
        
    """
    boxes = results["det"]
    masks = results.get("segment")
    h, w = source_image.shape[:2]
    # print("####",type(boxes.item()))
    boxes=[t.numpy() for t in boxes]
    persons = []
    person_centres = []
    violate = set()
    
    for i in range(len(boxes)):
            label = label_map[str(int(boxes[i][-1]))]
            if label=="person":
                x,y,w,h = tuple(boxes[i][:4])
                persons.append(tuple(boxes[i][:4]))
                person_centres.append([x+w//2,y+h//2])
    # print("##",len(persons))
    # print("$%%%",len(person_centres))
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
    
        
        tl = 1 or round(0.002 * (source_img.shape[0] + source_image.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(source_image,(int(x),int(y)),(int(w),int(h)),color,thickness=tl, lineType=cv2.LINE_AA)
        c1,c2=(int(x),int(y)),(int(w),int(h))
        if label=="person":
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(source_image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(source_image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
    cv2.putText(source_image,'Number of Violations : '+str(v),(80,source_image.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    return source_image
# Caluculate the euclidean distance
def dist(pt1,pt2):
    try:
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
    except:
        return
        
# Run the object detection
def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0, model="None", device="None"):
    player = None
    # if device != "CPU":
    #     model.reshape({0: [1, 3, 640, 640]})
    compiled_model = core.compile_model(model, device)
    try:
        # Create a video player to play with target fps.
        player = VideoPlayer(
            source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('../demo_videos/output.mp4', fourcc, 20.0, (1280,720))
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )

        processing_times = collections.deque()
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
            # Get the results.
            input_image = np.array(frame)
           
            start_time = time.time()
            # model expects RGB image, while video capturing in BGR
            detections = detect_without_preprocess(input_image, compiled_model)[0]
            stop_time = time.time()
            
            image_with_boxes = draw_results_new(detections, input_image, label_map)
            frame = image_with_boxes
                
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
                fontScale=f_width / 900,
                color=(0, 0, 255),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            
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
                key = cv2.waitKey(1)
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
# Start the video feed and run the object detection
run_object_detection(source="../data/pedestrians.mp4", flip=False, use_popup=True,skip_first_frames=0, model=yolov8n_with_preprocess_model, device="AUTO")