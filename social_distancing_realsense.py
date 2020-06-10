import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time
import random
from itertools import combinations
from statistics import mode, StatisticsError
import util_functions as uf
import csv

# test flag
test = False
plot = False
social_distance_viewer = False
personal_distance_viewer = False

# read balance file
balance = uf.read_balance_file('./balance.csv')
balance2 = uf.read_balance_file('./balance2.csv')

# colors
white = (255, 255, 255)
red = (0, 0, 255)
red_circle = (0, 0, 204)
green = (0, 255, 0)

# setup for darknet model
weightsPath = "./own_model/yolo-obj_65000.weights" #"./yolov3-tiny.weights"#
configPath = "./own_model/yolo-obj.cfg" #"./yolov3-tiny.cfg"#
width, height, fps = 1280, 720, 30 # optimal resolution

# realtime plotting
# matplotlib style
plt.style.use('ggplot')
plot_size = 50 # how many point visualize?
x_vec = np.linspace(0, 50, plot_size + 1)[0:-1]
y_vec = []
line1 = []
mean_line = []

# Filters pipe [Depth Frame >> Decimation Filter >> Depth2Disparity Transform** -> Spatial Filter >> Temporal Filter >> Disparity2Depth Transform** >> Hole Filling Filter >> Filtered Depth. ]
decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 2)
depth_to_disparity = rs.disparity_transform(True)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 3)
temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
disparity_to_depth = rs.disparity_transform(False)
hole_filling = rs.hole_filling_filter()

# setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) #1920, 1080

#Start Streaming
profile = pipeline.start(config)
dev = profile.get_device()

depth_sensor = dev.first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 3) # set high accuracy: https://github.com/IntelRealSense/librealsense/issues/2577#issuecomment-432137634

colorizer = rs.colorizer()
colorizer.set_option(rs.option.max_distance,15)#[0-16]
align = rs.align(rs.stream.color)

intrColor, intrDepth = uf.get_intrinsics(profile)

# DARKNET
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)


if __name__=='__main__':

	try:
		while True:
			st = time.time()

			#startt = time.time()
			for x in range(5):
				#Wait for pair of frames
				frame = pipeline.wait_for_frames()
			#end = time.time()
			#print( end - startt)

			for x in range(len(frame)):
				frame = decimation.process(frame).as_frameset()
				frame = depth_to_disparity.process(frame).as_frameset()
				frame = spatial.process(frame).as_frameset()
				frame = temporal.process(frame).as_frameset()
				frame = disparity_to_depth.process(frame).as_frameset()
				frame = hole_filling.process(frame).as_frameset()

			depth_frame = frame.get_depth_frame()
			color_frame = frame.get_color_frame()
			if not color_frame and not depth_frame:
				continue

			# align
			align = rs.align(rs.stream.color)
			frames = align.process(frame)

			aligned_depth_frame = frames.get_depth_frame()
			#Convert images to numpy arrays
			color_image = np.asanyarray(color_frame.get_data())
			colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

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


			distances = []
			distances_opt = []
			if len(idxs) > 0:
		            # flatten(): return a copy of the array collapsed into one dimension.
					for i in idxs.flatten():
						(x, y) = (boxes[i][0], boxes[i][1])
						(w, h) = (boxes[i][2], boxes[i][3])
						(centerX, centerY) = (boxes[i][4], boxes[i][5])

						#(torso_upperX, torso_upperY) = (centerX - (h/8), centerY + (h/8) + (h/16))
						#(torso_lowerX, torso_lowerY) = (centerX + (h/8), centerY - ((h/8) + (h/16)))
						(torso_upperX, torso_upperY) = (centerX - (h/8), centerY + (h/8))
						(torso_lowerX, torso_lowerY) = (centerX + (h/8), centerY - ((h/8)))

						# print shape only in test mode
						if(test):
							cv2.rectangle(colorized_depth, (x, y), (x + w, y + h), white, 2)
							# torso center
							cv2.circle(colorized_depth, (centerX,centerY), radius=4, color=red, thickness=-1)
							# torso rectangle
							cv2.rectangle(colorized_depth, (int(torso_upperX), int(torso_upperY)), (int(torso_lowerX), int(torso_lowerY)), red, 2)

						depth = np.asanyarray(aligned_depth_frame.get_data())
						# Crop depth data: !!!720x1280(YxX)!!!
						#print(int(torso_upperX), int(torso_lowerX), int(torso_lowerY), int(torso_upperY))

						depth = depth[int(torso_lowerY):int(torso_upperY), int(torso_upperX):int(torso_lowerX)].astype(float)
						#depth = depth[int(centerY):int(centerY+1), int(centerX):int(centerX+1)].astype(float)

						# Get data scale from the device and convert to meters
						depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
						depth = depth * depth_scale

						"""
						# remove data from depth matrix whether over a threshold (mode value)
						depth_mode = []
						try:
							mode_depth = mode(depth.flatten())
							[depth_mode.append(i) for i in depth.flatten() if(i<= mode_depth + 5. and i >= mode_depth - 5.)]
							z_axis = np.mean(depth_mode)
						except StatisticsError:
							print("No unique mode found")
						"""
						z_axis = np.mean(depth)

						try:
							text = "Distance: " + '{:0.2f}'.format(z_axis) + ' meters'
							x_opt, y_opt, z_opt = uf.convert_row_col_range_to_point(z_axis, h, w)
							text_opt = "Distance opt: " + '{:0.2f}'.format(uf.depth_optimized(z_axis, balance)) + ' meters'

							# [X, Y, Z, centerX, centerY]
							distances.append([uf.convert_depth_pixel_to_metric_coordinate(uf.depth_optimized(z_axis, balance), float(centerX), float(centerY), intrDepth), centerX, centerY])

							if(plot):
								if(len(y_vec) >= plot_size):
									y_vec[-1] = z_axis
									line1, mean_line = uf.live_plotter(x_vec, y_vec, line1, mean_line, "Realtime Z distance")
									y_vec = np.append(y_vec[1:],0.0)
								else:
									y_vec.append(z_axis)
								#uf.plot_data(z_axis, x_vec, y_vec, plot_size, line1, mean_line, "Realtime Z distance")

							distances_opt.append([(x_opt, y_opt ,z_opt), centerX, centerY])
						except RuntimeError:
							text = "Distance: NaN  meters"

						cv2.putText(colorized_depth, text_opt, (x,  y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)

						if(personal_distance_viewer):
							background = np.full((500,800,3), 125, dtype=np.uint8)
							cv2.putText(background, '{:0.2f}'.format(z_axis), (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 8.5, white, 2)
							cv2.imshow("Personal Distancing Analyzer", background)

						"""
						background = np.full((800,525,3), 125, dtype=np.uint8)
						cv2.rectangle(background, (20, 60), (510, 760), (170, 170, 170), 2)
						cv2.putText(background, "Analyzing warning distances", (20, 45),
									cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
						cv2.rectangle(background, (20, 60), (510, 760), (170, 170, 170), 2)
						cv2.putText(background, "A->B", (30, 80),
									cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
						cv2.putText(background, "C->D", (30, 100),
									cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
						cv2.imshow("Social Distancing Analyzer", background)
						"""

			if (len(distances) >= 2):
				# combinations of every bboxes found. Usare il coefficiente binomiale per calcolare le combinazioni rispetto ai bboxes trovati (54 bboxes = 1431)
				comb = combinations(distances, 2)
				for i in list(comb):
					social_distance = uf.euclidian_distance(i[0][:1], i[1][:1])
					#print(social_distance)

					# i[0][1] = center_X of the first element, i[0][2] = center_Y of the first element
					# i[1][1] = center_X of the second element, i[1][2] = center_Y of the second element
					if(social_distance <= 1.1):
						cv2.line(colorized_depth,(i[0][1], i[0][2]) , (i[1][1], i[1][2]) , red, 2)
						cv2.circle(colorized_depth, (i[0][1], i[0][2]), radius=4, color=red_circle, thickness=-1)
						cv2.circle(colorized_depth, (i[1][1], i[1][2]), radius=4, color=red_circle, thickness=-1)
					else:
						 if (test):
							 cv2.line(colorized_depth,(i[0][1], i[0][2]) , (i[1][1], i[1][2]) , green, 2)

					cv2.putText(colorized_depth, "social distance: " + '{:0.2f}'.format(social_distance), (i[0][1], i[0][2]) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)

					if(social_distance_viewer):
						background = np.full((800,525,3), 125, dtype=np.uint8)
						cv2.putText(background, '{:0.2f}'.format(social_distance), (10, 100) , cv2.FONT_HERSHEY_SIMPLEX, 3.5, white, 2)
						cv2.imshow("Social Distancing Analyzer", background)

			cv2.imshow("Social Distancing Viewer", colorized_depth)



			end = time.time()
			print("Frame Time (all routines) : {:.6f} seconds".format(end - st))

	finally:
		#Stop streaming
		pipeline.stop()
		cv2.destroyAllWindows()
