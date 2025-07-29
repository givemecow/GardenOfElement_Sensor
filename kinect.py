import pyk4a
from pyk4a import Config, PyK4A

class Kinect:

    def __init__(self):
        self.device = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            ))

