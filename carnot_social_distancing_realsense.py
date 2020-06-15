import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time

# https://medium.com/@manivannan_data/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
def getAngle(left, center, right):
	a = np.array(left)
	b = np.array(center)
	c = np.array(right)

	ba = a - b
	bc = c - b

	cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
	angle = np.arccos(cosine_angle)

	return np.degrees(angle)

def carnot(first, second, angle):
	return math.sqrt(((4 * np.square(first) + 4 * np.square(second)) - (8 * first * second * math.cos(angle))))**0.5



print("Environment Ready")
# https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb

labelsPath = "./own_model/data/obj.data"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


weightsPath = "./own_model/yolo-obj_90000.weights"
configPath = "./own_model/yolo-obj.cfg"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) #1920, 1080

#Start Streaming
profile = pipeline.start(config)

try:
	while True:

		#Wait for pair of frames
		frames = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		if not depth_frame or not color_frame:
			continue

		#Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		colorizer = rs.colorizer()
		colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

		#Apply colormap on depth image (image must be converted to 8-bit)
		#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

		# align
		align = rs.align(rs.stream.color)
		frames = align.process(frames)

		aligned_depth_frame = frames.get_depth_frame()
		colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

		#Stack images horizontally
		#images = np.hstack((color_image, depth_colormap))


		#Show images
		cv2.namedWindow('Social Distancing', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('Social Distancing', colorized_depth)
		#cv2.waitKey(1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		# DARKNET
		net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

		# retrieve height and with of the frame
		(H, W) = (color_frame.get_height(), color_frame.get_width())

		# return neural network configuration
		ln = net.getLayerNames()
		# find all unconnectedOutLayers
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		# resize image (416*416)
		blob = cv2.dnn.blobFromImage(color_image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		# set new frame (resized)
		net.setInput(blob)
		start = time.time()
		# Runs forward pass to compute output of layer
		layerOutputs = net.forward(ln)
		end = time.time()
		print("Frame Prediction Time : {:.6f} seconds".format(end - start))

		boxes = []
		confidences = []
		classIDs = []
		for output in layerOutputs:
			for detection in output:

				scores = detection[5:]
				# classID -> 0 = student
				classID = np.argmax(scores)
				# confidence
				confidence = scores[classID]

				if confidence > 0.5 and classID == 0:

					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					# for every frame: generate bboxes, confidence and classID for all people detected
					boxes.append([x, y, int(width), int(height), int(centerX), int(centerY)])
					confidences.append(float(confidence))
					classIDs.append(classID)

	    # Performs non maximum suppression given boxes and corresponding scores
	    # https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
	    # Help us to generate only one bbox for every people in the frame
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
		ind = []
		for i in range(0,len(classIDs)):
			if(classIDs[i]==0):
				ind.append(i)


		# white
		color = (255, 255, 255)

		center_bbox = []
		distances = []
		if len(idxs) > 0:
	            # flatten(): return a copy of the array collapsed into one dimension.
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					(centerX, centerY) = (boxes[i][4], boxes[i][5])
					center_bbox.append([centerX, centerY])

					cv2.rectangle(colorized_depth, (x, y), (x + w, y + h), color, 2)

					# line from the center of the camera
					cv2.line(colorized_depth, (int(W/2), H) , (int(centerX), int(centerY)) , color, 2)
					# torso
					cv2.circle(colorized_depth, (centerX,centerY), radius=4, color=(255, 255, 255), thickness=-1)
					(torso_upperX, torso_upperY)= (centerX - (h/8), centerY + (h/8) + (h/16))
					(torso_lowerX, torso_lowerY)= (centerX + (h/8), centerY - ((h/8) + (h/16)))
					cv2.rectangle(colorized_depth, (int(torso_upperX), int(torso_upperY)), (int(torso_lowerX), int(torso_lowerY)), color, 2)


					depth = np.asanyarray(aligned_depth_frame.get_data())
					# Crop depth data: !!!720x1280(YxX)!!!
					#print(int(torso_upperX), int(torso_lowerX), int(torso_lowerY), int(torso_upperY))
					depth = depth[int(torso_lowerY):int(torso_upperY), int(torso_upperX):int(torso_lowerX)].astype(float)
					#print(depth)
					# Get data scale from the device and convert to meters
					depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
					depth = depth * depth_scale
					dist,_,_,_ = cv2.mean(depth)
					distances.append(dist)

					try:
						text = "Distance: " + str(dist) + ' meters'
						print("distance: " + str(dist))
					except RuntimeError:
						text = "Distance: NaN  meters"
					cv2.putText(colorized_depth, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

		# per ora funge solo con due bboxes, TODO: implemetare per qualsiasi numero di bboxes
		if (len(center_bbox) == 2):
			angle = getAngle((center_bbox[0][0], center_bbox[0][1]), (int(W/2), H), (center_bbox[1][0], center_bbox[1][1]))
			print("angle: " + str(angle))
			social_distance = carnot(distances[0], distances[1], angle)
			cv2.line(colorized_depth,(center_bbox[0][0], center_bbox[0][1]) , (center_bbox[1][0], center_bbox[1][1]) , color, 2)
			cv2.putText(colorized_depth, "social distance: " + str(social_distance), (40, 40) , cv2.FONT_HERSHEY_SIMPLEX,1.5, color, 2)

		cv2.imshow("Social Distancing ", colorized_depth)

finally:

	#Stop streaming
	pipeline.stop()
