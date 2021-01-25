import numpy as np
import cv2
import argparse
import matplotlib as plt
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-d", "--device", required=False,
	help="Device number")
args = vars(ap.parse_args())


DEVICE = int(args["device"]) or 0
cap = cv2.VideoCapture(DEVICE)

# MODEL = r"/Users/millyduong/Documents/yolo_face/face_rec/yolo/drive-download-20210121T071505Z-001/yolov3-face.cfg"
# WEIGHT = r"/Users/millyduong/Documents/yolo_face/face_rec/yolo/drive-download-20210121T071505Z-001/yolov3-wider_16000.weights"

dirname = os.path.dirname(__file__)
MODEL = os.path.join(dirname, 'yolo', 'yolov3-face.cfg')
WEIGHT = os.path.join(dirname, 'yolo', 'yolov3-wider_16000.weights')



net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

IMG_WIDTH, IMG_HEIGHT = 416, 416

total = 0
while(cap.isOpened()):
    ret, frame = cap.read()
   

    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)
                                

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)


    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.

    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
            # 1 out has multiple predictions with length of 6
        for detection in out:
            confidence = detection[-1]
                    # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                
                            # Find the top left point of the bounding box 
                topleft_x = center_x - width/2
                topleft_y = center_y - height/2
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    result = frame.copy()
    final_boxes = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 255, 0) 
    thickness = 2
    key = cv2.waitKey(1) & 0xFF
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)

        # Extract position data
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])

        # Draw bouding box with the above measurements
        ### YOUR CODE HERE
        cv2.rectangle(result, (left,top), ((left+width), (top+height)), (0,255,0), 2)
        
    #     image = cv2.rectangle(image, start_point, end_point, color, thickness) 
        face = frame[top:top+height,left:left+width]
            
            # Display text about confidence rate above each box
        text = f'{confidences[i]:.2f}'
        ### YOUR CODE HERE

        cv2.putText(result, text, (left,top-5), font, fontScale, color,    thickness)
        # if the `k` key was pressed, write the *original* frame to disk
        # so we can later process it and use it for face recognition
        if key == ord("k"):
            p = os.path.sep.join([args["output"], "{}.png".format(
                str(total).zfill(5))])
            cv2.imwrite(p, face)
            total += 1

    # Display text about number of detected faces on topleft corner
    # YOUR CODE HERE
    
    cv2.putText(result,
                f"Number of faces:{len(indices)}", (10, 50), font, fontScale + 1, color, thickness)

    cv2.imshow("Frame", result)

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break


print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
