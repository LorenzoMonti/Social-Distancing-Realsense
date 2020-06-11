import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time
from itertools import combinations
from statistics import mode, StatisticsError
import util_functions as uf
import csv

# test flag
filter = True

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


pipeline = rs.pipeline()
config = rs.config()
width, height, fps = 1280, 720, 30 # optimal resolution
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) #1920, 1080

#Start Streaming
profile = pipeline.start(config)
dev = profile.get_device()

depth_sensor = dev.first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 3) # set high accuracy: https://github.com/IntelRealSense/librealsense/issues/2577#issuecomment-432137634

colorizer = rs.colorizer()
colorizer.set_option(rs.option.max_distance,15)#[0-16]

region_of_interest = []
roiSelected = False
z_array = []

try:
    while True:
        st = time.time()
        if(filter):
            for x in range(5):
                frame = pipeline.wait_for_frames()

            for x in range(len(frame)):
                frame = decimation.process(frame).as_frameset()
                frame = depth_to_disparity.process(frame).as_frameset()
                frame = spatial.process(frame).as_frameset()
                frame = temporal.process(frame).as_frameset()
                frame = disparity_to_depth.process(frame).as_frameset()
                frame = hole_filling.process(frame).as_frameset()

        else:
            frame = pipeline.wait_for_frames()

        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        #Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # align
        align = rs.align(rs.stream.color)
        frame = align.process(frame)

        aligned_depth_frame = frame.get_depth_frame()
        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())


        key = cv2.waitKey(1) & 0xFF

        # quit
        if key == ord('q'):
            break

        # pause
        if key == ord('p'):
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow("RGB Viewer", color_image)
                cv2.imshow("Depth Viewer", colorized_depth)

                # x,y,w,h
                region_of_interest = cv2.selectROI("RGB Viewer", color_image, False, False)
                #print(region_of_interest)
                roiSelected = True
                break

                if key2 == ord('p'):
                    break

        if key == ord('s'):
            print("enter real distance: ")
            real = input()
            file = open("distance_filter.csv", "a")
            file.write(str(real) + "," + str(np.mean(z_array)) + "\n")
            file.close()

        cv2.imshow("RGB Viewer", color_image)
        if(roiSelected):

            (x,y,w,h) = region_of_interest
            cv2.rectangle(colorized_depth, (x, y), (x + w, y + h), (255, 255, 255), 2)

            depth = np.asanyarray(aligned_depth_frame.get_data())

            # selected ROI
            depth = depth[int(y):int(y) + int(h), int(x):int(x) + int(w)].astype(float)

            # center, top left, top right, bottom left and bottom right of selected ROI
            #depthCenter      = depth[(int(x) + (int(w / 2)), int(y) + (int(h / 2)))].astype(float)
            #depthTopLeft     = depth[(int(x), int(y))].astype(float)
            #depthTopRight    = depth[(int(x) + (int(w)), int(y))].astype(float)
            #depthBottomLeft  = depth[(int(x), int(y) + int(h))].astype(float)
            #depthBottomRight = depth[(int(x) + int(w), int(y) + int(h))].astype(float)
            #cv2.imshow("Cropped Viewer", depth)
            #depth = depthTopLeft  + depthCenter + depthBottomLeft + depthBottomRight

            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth = depth * depth_scale
            cv2.circle(colorized_depth, (int(x) + (int(w / 2)), int(y) + (int(h / 2))), radius=4, color=(0,0,255), thickness=-1)
            cv2.circle(colorized_depth, (int(x), int(y)), radius=4, color=(0,0,255), thickness=-1)
            cv2.circle(colorized_depth, (int(x) + (int(w)), int(y)), radius=4, color=(0,0,255), thickness=-1)
            cv2.circle(colorized_depth, (int(x), int(y) + int(h)), radius=4, color=(0,0,255), thickness=-1)
            cv2.circle(colorized_depth, (int(x) + int(w), int(y) + int(h)), radius=4, color=(0,0,255), thickness=-1)

            z_axis = np.average(depth)
            z_array.append(z_axis)
            #print(np.mean(z_array))
            #print(len(z_array))
            background = np.full((305,1200,3), 125, dtype=np.uint8)
            #cv2.rectangle(background, (20, 60), (295, 480), (170, 170, 170), 2)
            cv2.putText(background, 'mean z axis: {:0.6f}'.format(np.mean(z_array)), (10,  250), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 2)
            cv2.putText(background, '# iterations: {}'.format(len(z_array)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 2)
            cv2.imshow("Z analyzer", background)

        cv2.imshow("Depth Viewer", colorized_depth)

        end = time.time()
        print("Frame Time: {:.6f} seconds".format(end - st))
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
