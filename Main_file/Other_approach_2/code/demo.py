# Import OpenVINO modules
import openvino
from openvino.inference_engine import IECore, IENetwork
import openvino.runtime as ov
import time

# Load the IR model files
model_xml = "../models/person-detection-0202/FP32/person-detection-0202.xml"
model_bin = "../models/person-detection-0202/FP32/person-detection-0202.bin"

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)

# Get the input and output layer names
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs)) 

# Load the network to the device (CPU, GPU, etc.)
core = ov.Core()
model = core.compile_model(model_xml,"CPU")

def dist(pt1,pt2):
    try:
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
    except:
        return


# Define a function to compute the Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Define a threshold for the minimum distance between people
distance_threshold = 200 # pixels

# Read and preprocess the input video
import cv2
import numpy as np

#give the input video
video = cv2.VideoCapture("../data/pedestrians.mp4")
_,frame = video.read()
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"MJPG")

#give the output address to store the video 
writer = cv2.VideoWriter('../demo_videos/output.avi', fourcc, 30,(width,height), True)

# used to record the time when we processed last frame
prev_frame_time = 0

distance_thres=100
# used to record the time at which we processed current frame
new_frame_time = 0
 
while True:
    # Read a frame from the video.
    ret, frame = video.read()
    
    if not ret:
        break # Exit the loop if end of video or error
    height, width = frame.shape[:2]
   
    # Preprocess the frame
    image = cv2.resize(frame, (512,512))
    image = image.transpose((2, 0, 1)) # Change data layout from HWC to CHW
    
    

    # Run inference and get the output
    infer_request = model.create_infer_request()
    input_shape = [1,3,512,512]   
    input_tensor= ov.Tensor(image.astype(np.float32))
    input_tensor.shape = input_shape
    infer_request.set_tensor(input_blob,input_tensor)
    infer_request.start_async()
    infer_request.wait()
    output_tensor = infer_request.get_tensor(output_blob)
    output = output_tensor.data
    

    # Parse the output and get the bounding boxes of detected people
    boxes = []
    confidences = []
    class_ids = []
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    
    for detection in output[0][0]:
        # Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max]
        if detection[2] > 0.5: # Only keep detections with confidence > 0.5
            class_id = int(detection[1])
            if class_id == 0: # Only keep detections with label 0 (person)
                x_min = int(detection[3] * width)
                y_min = int(detection[4] * height)
                x_max = int(detection[5] * width)
                y_max = int(detection[6] * height)
                boxes.append([x_min, y_min, x_max, y_max])
                confidences.append(float(detection[2]))
                class_ids.append(class_id)
    
    n = len(boxes)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    label="person"
    

    v = 0
    person_centers=[]
    persons=[]
    violate=set()
    for i in indices:
        box = boxes[i]
        # Draw a bounding box around the person
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # Get the center point of the box
        center_a = np.array([box[0] + (box[2] - box[0]) / 2, box[1] + (box[3] - box[1]) / 2])
        persons.append(boxes[i])
        person_centers.append(center_a)

    for i in range(len(persons)):
        for j in range(i+1,len(persons)):
            distance = euclidean_distance(person_centers[i], person_centers[j])
            if distance < distance_threshold:
                violate.add(tuple(persons[i]))
                violate.add(tuple(persons[j]))
    
    for (x,y,w,h) in persons:
            # print((x,y,w,h))
            if tuple((x,y,w,h)) in violate:
                    color = (0,0,225)
                    v+=1
                    # print("YES")
            else:
                    
                    color = (125,255,0)
    
            cv2.rectangle(frame,(x,y),(w,h),color,2)
            tl = 2 or round(0.002 * (frame.shape[0] + source_image.shape[1]) / 2) + 1  # line/font thickness
            color = color or [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(frame,(int(x),int(y)),(int(w),int(h)),color,thickness=tl, lineType=cv2.LINE_AA)
            c1,c2=(int(x),int(y)),(int(w),int(h))
            if label=="person":
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            
    # # Show the output frame
    cv2.namedWindow("Social Distance Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Social Distance Detector", 800, 600)
    cv2.putText(frame,'Number of Violations : '+str(v),(80,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    
    cv2.putText(frame,"FPS :"+ fps, (7,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3, cv2.LINE_AA)
    cv2.imshow("Social Distance Detector", frame)
    writer.write(frame)
    
    
    
    
    # Wait for a key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video and destroy the windows
video.release()
writer.release()
cv2.destroyAllWindows()
