__all__ = ['setGImages', 'getGImages', 'Images']

import numpy as np
from imutils import resize

from .horizon import *
from .image_utils import min_pool, BH_op
import time, sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32":
    # On Windows, the best timer is time.clock()
    default_timer = time.perf_counter
else:
    # On most other platforms the best timer is time.time()
    default_timer = time.time

import logging

logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




def get_project_root() -> Path:
    return Path(__file__).parent.parent


@dataclass
class Images:
    maxpool: int = 12
    # CMO_kernalsize = 3
    full_rgb: np.array = None
    # small_rgb: np.array = None
    full_gray: np.array = None
    small_gray: np.array = None
    minpool: np.array = None
    # minpool_f: np.array = None
    last_minpool_f: np.array = None
    cmo: np.array = None
    mask: np.array = None
    horizon: np.array = None
    file_path = None

    def set(self, image: np.array, _file_path: str = ''):
        self.full_rgb = image
        self.file_path = _file_path
        if self.full_rgb.ndim == 3:
            # use this as much faster than cv2.cvtColor(imgrgb, cv2.COLOR_BGR2GRAY) (~24msec for 6K image)
            self.full_gray = self.full_rgb[:, :, 1]

        self.minpool = min_pool(self.full_gray, self.maxpool, self.maxpool)
        small_gray = resize(self.full_gray, width=self.minpool.shape[1])
        # self.small_rgb = resize(self.full_rgb, width=self.minpool.shape[1])
        self.small_gray = np.zeros_like(self.minpool, dtype='uint8')
        n_rows = min(self.minpool.shape[0], small_gray.shape[0])
        self.small_gray[:n_rows, :] = small_gray[:n_rows, :]
        # self.small_gray = small_gray
        # self.minpool_f = np.float32(self.minpool)

    def mask_sky(self):
        self.mask = find_sky_2(self.minpool, threshold=80, kernal_size=7)



# Set global images buffer
g_images = Images()
g_images.small_rgb = np.random.normal(size=(320, 480, 3), loc=1024, scale=64).astype(np.uint16)


def setGImages(image, file_path=None):
    global g_images
    g_images.set(image, file_path)


def getGImages() -> Images:
    global g_images
    return g_images


