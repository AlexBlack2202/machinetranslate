# import the necessary packages
import numpy as np
import argparse
import cv2
import glob

 

model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt_path =  "deploy.prototxt.txt"

confi = 0.5

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it


image_names = glob.glob("dimg/*.jpg")
for image_name in image_names:
	#image_name = "dimg/a.jpg"
	image = cv2.imread(image_name)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()
	count = 0

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
		
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > confi:
			count += 1
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}".format(confidence) + ' - ' + str(count)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
	cv2.putText(image, str(count)+" face", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	cv2.imwrite("pimg/"+image_name,image)
	# show the output image
	#cv2.imshow("Output", image)
	#cv2.waitKey(0)
