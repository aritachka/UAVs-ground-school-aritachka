#!/usr/bin/env python3
import cv2
import numpy as np

cap = cv2.VideoCapture("Minecraft_stitch_test.mp4")
sift = cv2.SIFT_create()
matcher = cv2.BFMatcher()

frame_interval = 60
frame_count = 0
matches_number = 1000

#get first frame of video
ret, stiched = cap.read()

while True:
	#downsampling
	if not frame_count % frame_interval == 0:
		frame_count += 1
		ret, frame = cap.read()
		continue
	frame_count += 1
	
	#get matches
	ret, frame = cap.read()
	kp1, des1 = sift.detectAndCompute(stiched, None)
	kp2, des2 = sift.detectAndCompute(frame, None)
	matches = matcher.knnMatch(des1, des2, k=2)
	
	#filter matches
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])
	
	#get coordinates of matches
	first_run = True
	for mat in good[0:matches_number]:
		mat = mat[0]
		if first_run == True:
			stiched_coords = np.array([kp1[mat.queryIdx].pt])
			frame_coords = np.array([kp2[mat.trainIdx].pt])
			first_run = False
			continue

		stiched_coords = np.vstack((stiched_coords, np.array([kp1[mat.queryIdx].pt])))
		frame_coords = np.vstack((frame_coords, np.array([kp2[mat.trainIdx].pt])))
	print(stiched_coords)
	print(frame_coords)
	#find transformation
	"""
	H = cv2.findHomography(frame_coords, stiched_coords, cv2.RANSAC, 2.0)
	M = H[0]

	height, width = stiched.shape[:2]
	aligned_frame = cv2.warpPerspective(frame, M, (width, height))
	cv2.imshow('aligned frame', aligned_frame)
	"""
	#H = cv2.getAffineTransform(frame_coords, stiched_coords)
	H, inliers = cv2.estimateAffinePartial2D(frame_coords, stiched_coords)

	stiched_height, stiched_width = stiched.shape[:2]
	frame_height, frame_width = frame.shape[:2]
	stiched_corners = np.array([
		[0, 0],
		[stiched_width, 0],
		[stiched_width, stiched_height],
		[0, stiched_height],
	])
	frame_corners = np.array([
		[0, 0],
		[frame_width, 0],
		[frame_width, frame_height],
		[0, frame_height],
	])
	print(frame_corners)
	new_corners = cv2.transform(np.array([frame_corners]), H)
	print(new_corners)
	all_corners = np.vstack((new_corners[0], stiched_corners))
	print(all_corners)
	[xmin, ymin] = all_corners.min(axis=0).flatten()
	[xmax, ymax] = all_corners.max(axis=0).flatten()
	print([xmin, ymin])
	print([xmax, ymax])
	
	translation = [-xmin, -ymin]
	H_translation = np.array([[1, 0, -xmin], [0, 1, -ymin]], dtype=np.float32)
	
	# Step 3: Warp img2 onto the canvas
	print(H_translation)
	print(H)
	
	
	H_translation = np.vstack((H_translation, [0, 0, 1]))
	H = np.vstack((H, [0, 0, 1]))
	transformation_matrix = H_translation @ H
	transformation_matrix = transformation_matrix[:2]
	print(transformation_matrix)
	stitched_img = cv2.warpAffine(frame, transformation_matrix, (xmax-xmin, ymax-ymin))
	stitched_img[translation[1]:stiched_height+translation[1], translation[0]:stiched_width+translation[0]] = stiched
		
	#stiched_canvas = cv2.copyMakeBorder(stiched, 100, 0, 100, 100, cv2.BORDER_CONSTANT, value=(0, 0, 0))
	#height, width = stiched_canvas.shape[:2]
	#print(height, width)
	#aligned_frame = cv2.warpAffine(frame, H, (width, height))
	cv2.imshow('framebruh', stitched_img)
	
	"""
	print(H)
	top = 0
	bottom = 0
	left = 0
	right = 0
	
	if H[0][2] < 0:
		left = width*abs(H[0][2])
	else:
		right = width*H[0][2]
	
	if H[1][2] < 0:
		top = height*abs(H[1][2])
	else:
		bottom = height*H[1][2]
		
	
	print(top, bottom, right, left)
	"""

	#img3 = cv2.drawMatchesKnn(stiched,kp1,frame,kp2,good[0:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	
	alpha = 0.5
	#stiched = cv2.addWeighted(stiched_canvas, 1 - alpha, aligned_frame, alpha, 0)
	cv2.imshow('frame', stiched)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	