import numpy as np
import cv2 
import argparse
import sys
import os.path

import os

folder = './data/processed'
confidence_threshold = 0.5
nms_threshold = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

arg = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
arg.add_argument('--image', help='Path to image file')
arg.add_argument('--video', help='Path to video file')

args = arg.parse_args()

class_names = 'coco.names'
classes = None
with open(class_names, 'rt') as f:
	classes = f.read().rstrip('\n').split('\n')

print('Number of classes:', len(classes))
#print(classes)

MODEL = './yolo/yolov3-face.cfg'
WEIGHT = './yolo/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_output_names(net):
	layers_names = net.getLayerNames()
	#print('full:', len(layers_names))
	#print(layers_names)
	return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def extract_faces(frame, left, top, right, bottom):
	"""
	return the face portion of an image
	"""
	copy = frame.copy()	
	top = max(0, top-20)
	bottom = min(bottom+20, frame.shape[0])
	left = max(0, left-20)
	right = min(frame.shape[1], right+20)
	#face = copy[top-20: bottom + 20, left-20: right+20]
	face = copy[top: bottom, left: right]	
	#cv2.imwrite(output_file, face.astype(np.uint8))
	return face


def draw_prediction(class_id, confidence, left, top, right, bottom):
	face = extract_faces(frame, left, top, right, bottom)
	
	# Draw a bounding box.
	cv2.rectangle(frame, (left - 20, top - 20), (right + 20, bottom + 20), (0, 255, 0), 2)
	label = '%.2f' % confidence
	# Get the label for the class name and its confidence
	if classes:
		assert (class_id < len(classes))
		label = '%s:%s' % (classes[class_id], label)
	# Display the label at the top of the bounding box
	labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, labelSize[1])		
	return face


def post_process(frame, outs):
	frame_height = frame.shape[0]
	frame_width = frame.shape[1]
	
	class_ids = []
	confidences = []
	boxes = []
	extracted_faces = []
	for out in outs:
		for detection in out:
			scores = detection[5:]	
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > confidence_threshold:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				center_x = int(detection[0] * frame_width)
				center_y = int(detection[1] * frame_height)
				width = int(detection[2] * frame_width)
				height = int(detection[3] * frame_height)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				class_ids.append(class_id)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])

	indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)	
	print(len(indices))
	for i in indices:
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		extracted_face = draw_prediction(class_ids[i], confidences[i], left, top, left + width, top + height)
		extracted_faces.append(extracted_face)
	
	return extracted_faces


temp_ls = get_output_names(net)
#print('Output layer:', temp_ls)



if __name__ == '__main__':
	window = 'Object Detection in OpenCV'
	cv2.namedWindow(window, cv2.WINDOW_NORMAL)

	output_file = 'yolo_out.avi'

	if args.image:
		if not os.path.isfile(args.image):
			print("Input image file ", args.image, " doesn't exist")
			sys.exit(1)	

		cap = cv2.VideoCapture(args.image)
		output_file = args.image[:-4] + 'yolo_out.jpg'
	
	elif args.video:
		if not os.path.isfile(args.video):
			print("Input video file ", args.video, " doesn't exist")
			sys.exit(1)	

		cap = cv2.VideoCapture(args.video)
		output_file = args.video[:-4] + 'yolo_out.avi'
	
	else:
		cap = cv2.VideoCapture(0)

	
	# Get the video writer initialized to save the output video
	if not args.image:
		vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
									(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

	while cv2.waitKey(1) < 0:
		
		ret, frame = cap.read()
		if not ret:
			print("Done processing !!!")
			print("Output file is stored as ", output_file)
			cv2.waitKey(3000)
			# Release device
			cap.release()
			break	

		# Create 4D blob from a frame
		blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0,0,0], 1, crop=False)
		# Set the input to the network	
		net.setInput(blob)
		# Runs the forward pass to get the output of the output layers
		outs = net.forward(get_output_names(net))
		# Remove the bounding boxes with low confidence	
		extracted_faces = post_process(frame, outs)
		print('Detected {} face'.format(len(extracted_faces)))

		# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and
		# the timings for each of the layers(in layersTimes)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
		cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

		# Write the frame with the detection boxes
		#print('Outputfile:', output_file)
		if args.image:
			#cv2.imwrite(output_file, frame.astype(np.uint8))
			#for index, face in enumerate(extracted_faces):
			#	filename = 'face'	
			face = extracted_faces[0]
			cv2.imwrite(output_file, face.astype(np.uint8))

		else:
			vid_writer.write(frame.astype(np.uint8))

		
		cv2.imshow(window , frame)
