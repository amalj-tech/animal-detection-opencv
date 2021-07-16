# Test this script:
# python3 detect-animal-from-img.py --image images/1horse.jpg --yolo yolo-coco
# python3 detect-animal-from-img.py --image images/1dog1cat.jpg --yolo yolo-coco
# python3 detect-animal-from-img.py --image images/2cats2dogs.jpg --yolo yolo-coco
# python3 detect-animal-from-img.py --image images/3cats3dogs.jpg --yolo yolo-coco
# python3 detect-animal-from-img.py --image images/5elephants.jpg --yolo yolo-coco
# python3 detect-animal-from-img.py --image images/5elephants.jpg --yolo yolo-coco
#Negative testing:
#Failed:# python3 detect-animal-from-img.py --image images/dining_table.jpg --yolo yolo-coco
#Success:# python3 detect-animal-from-img.py --image images/brick_wall.jpg --yolo yolo-coco

# import the necessary packages
import argparse
import os
import time
import cv2
import numpy as np
# Make sure necessary arguments are passed to the script
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

print("Number of detections:", len(idxs))
## print(type(idxs)) <class 'numpy.ndarray'>

# ensure at least one detection exists
if len(idxs) > 0:
	if len(idxs) == 1:
		print("Only 1 animal:")
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# print(type(i)) <class 'numpy.int32'>
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
			break;

		# Detection Output here
		print("Detected : ", text)

	elif len(idxs) > 1:
		# loop over the indexes we are keeping
		uniq_animals_set = ()
		uniq_animals_dict = {}
		multi_animal_counter = 1
		for i in idxs.flatten():
			# print(type(i)) <class 'numpy.int32'>
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			curr_text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			curr_animal = str(LABELS[classIDs[i]])
			curr_confidence = float("{:.4f}".format(confidences[i]))
			if curr_animal not in uniq_animals_set :
				uniq_animals_set = uniq_animals_set + (curr_animal,)
				uniq_animals_dict[curr_animal] = 1
			else:
				uniq_animals_dict[curr_animal] = uniq_animals_dict[curr_animal] + 1
			print("Animal{} : {}".format(multi_animal_counter,curr_text))
			cv2.putText(image, curr_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			# Increase the counter
			multi_animal_counter += 1

		# Detection Output here
		for anml in uniq_animals_set :
			print("Total count of {} is {}".format(anml,uniq_animals_dict[anml]))
else:
	print("No detection")

# show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)