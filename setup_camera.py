import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
from constants import *
import util_functions as uf

class SetupCamera:

    def __init__(self, camera):
        self.camera = camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = 0
        self.dev = profile.get_device()
        self.colorizer = 0
        self.align = 0

    def setup_camera():
        # setup
        config.enable_device(self.camera)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) #1920, 1080

        #Start Streaming
        profile = pipeline.start(config)

        depth_sensor = dev.first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3) # set high accuracy: https://github.com/IntelRealSense/librealsense/issues/2577#issuecomment-432137634
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.max_distance,15)#[0-16]
        align = rs.align(rs.stream.color)
