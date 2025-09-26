#!/usr/bin/env python3 PLEASE IGNORE, test file
import cv2
import numpy as np

cap = cv2.VideoCapture("Minecraft_stitch_test.mp4")
sift = cv2.SIFT_create()
orb = cv2.ORB_create()
matcher = cv2.BFMatcher()

frame_interval = 150
frame_count = 0
matches_number = 10000
images = []

while True:
	#downsampling
	if not frame_count % frame_interval == 0:
		frame_count += 1
		ret, frame = cap.read()
		if ret == False:
			break
		continue
	frame_count += 1
	
	#get matches
	ret, frame = cap.read()
	if ret == False:
		break
	images.append(frame)
	print('appended image')
	print(frame_count)

print(images)
stitcher = cv2.Stitcher.create()
print('test')
status, pano = stitcher.stitch(images)
print('hello')
cv2.imshow('panorama', pano)
cv2.waitKey(0)
cv2.destroyAllWindows()
	
