import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time



def get_intrinsics(profile):
	# Get intrinsics
	depth_sensor = profile.get_device().first_depth_sensor()
	cam_scale = depth_sensor.get_depth_scale()
	intrc =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
	intrd =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
	return intrc, intrd

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
				Euclidian distance method (3D)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
Convert the depth and image point information to metric coordinates
Parameters:
-----------
depth 	 	 	 : double
					   The depth value of the image point
pixel_x 	  	 	 : double
					   The x value of the image coordinate
pixel_y 	  	 	 : double
						The y value of the image coordinate
camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
Return:
----------
X : double
	The x value in meters
Y : double
	The y value in meters
Z : double
	The z value in meters
"""
def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
	X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx *depth
	Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy *depth
	return X, Y, depth

"""
Get Euclidian distance between two 3D points
-----------
point1 	 	 	 	: list(tuple)
point2 	  	 	 : list(tuple)

Return:
----------
euclidian distance : double

"""
def euclidian_distance(point1, point2):
	point1_X, point1_Y, point1_Z = point1[0]
	point2_X, point2_Y, point2_Z = point2[0]
	return(math.sqrt( (point2_X - point1_X)**2 + (point2_Y - point1_Y)**2 + (point2_Z - point1_Z)**2 ))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
						Carnot Method (2D)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
	return math.sqrt(((np.square(first) + np.square(second)) - (2 * first * second * math.cos(angle))))
