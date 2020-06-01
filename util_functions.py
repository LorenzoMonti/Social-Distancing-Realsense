import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time


# Constants
COLS = 1280
ROWS = 720

HFOV = 1.18997230395261 # 1.18124533204579
VFOV = 0.726878003768506 # 0.72083537591355
VCENTRE = (ROWS - 1) / float(2)
HCENTRE = (COLS -1) / float(2)
A = -3.38769422283892e-06 # -6.74058700563481e-06
B = -0.0039721384171171 # 0.0162142413358919
C = 1.35603974925075 # -15.1823712120162

VSIZE = math.tan(VFOV / 2) * 2
HSIZE = math.tan(VFOV / 2) * 2
VPIXEL = VSIZE / (ROWS - 1)
HPIXEL = HSIZE / (COLS - 1)


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
	X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx * depth
	Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy * depth
	return X, Y, depth

def read_balance_file():
	balance = []
	with open('./balance.csv', 'r') as csvfile:
		for line in csvfile.readlines():
			col = line.split(',')
			balance.append([float(col[0]),float(col[1])])
	return balance

def depth_optimized(depth, balance):
	index = 0
	for i in balance:
		if (depth >= i[0]):
			index = i
	depth = depth + index[1]
	return depth

def convert_row_col_range_to_point(depth, row, col):
	vratio = (VCENTRE - row) * VPIXEL
	hratio = (col - HCENTRE) * HPIXEL
	z = depth + A * depth * depth + B * depth #+ C
	y = depth * (-vratio)
	x = depth * hratio
	return x, y, z

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
