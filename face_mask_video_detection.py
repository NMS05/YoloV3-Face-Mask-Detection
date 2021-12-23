# do all the necessary imports
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet


# converting BB coordinates in yolo txt format in pixels 
def convertBack(x, y, w, h):
	xmin = int(round(x - (w / 2)))
	xmax = int(round(x + (w / 2)))
	ymin = int(round(y - (h / 2)))
	ymax = int(round(y + (h / 2)))
	return xmin, ymin, xmax, ymax


# drawing bounding boxes in image from detections
def cvDrawBoxes(detections, img,dim):
	for detection in detections:
		x, y, w, h = int((detection[2][0]/dim)*width),\
			int((detection[2][1]/dim)*height),\
			int((detection[2][2]/dim)*width),\
			int((detection[2][3]/dim)*height)
		xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
		pt1 = (xmin, ymin)
		pt2 = (xmax, ymax)      
		cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
		string = detection[0].decode()
		cv2.putText(img, string + ":" + str(round(detection[1],2)),(pt1[0]-5, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,[0, 0, 255], 2)

	return img

# define and load the trained models in GPU's"
mask_configPath = "face_mask/face_mask.cfg"
mask_weightPath = "face_mask/face_mask_final.weights" 
mask_metaPath = "face_mask/face_mask.data"   

darknet.set_gpu(0)
    
mask_netMain = darknet.load_net_custom(mask_configPath.encode("ascii"), mask_weightPath.encode("ascii"), 0, 1)
mask_metaMain = darknet.load_meta(mask_metaPath.encode("ascii"))

# initialize video
cap = cv2.VideoCapture(<video_file_name.mp4>)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
out = cv2.VideoWriter(<video_out.mp4>, cv2.VideoWriter_fourcc(*"MJPG"), 20.0,(width,height))


while True:

	ret,frame = cap.read()
	if ret==False: break

	# Opencv uses BGR format but Darknet uses RGB format
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Variable to store image
	darknet_image = darknet.make_image(darknet.network_width(mask_netMain),darknet.network_height(mask_netMain),3)

	# Resise image to 416x416 and copy to the above variable
	frame_resized = cv2.resize(frame_rgb,(darknet.network_width(mask_netMain),darknet.network_height(mask_netMain)),interpolation=cv2.INTER_LINEAR)
	darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

	# YoloV3 detections
	mask_detections = darknet.detect_image(mask_netMain, mask_metaMain, darknet_image, thresh=0.3)

	# Draw Bounding Boxes
	image = cvDrawBoxes(mask_detections,frame,416)

	# show and save detection results
	cv2.imshow('face_mask', image)
	cv2.waitKey(50)
	out.write(image)

cap.release()
out.release()
cv2.destroyAllWindows()
