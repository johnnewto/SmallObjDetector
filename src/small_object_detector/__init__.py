"""
This module provides various utilities for small object detection.

Classes:
- ImageLoader: Loads images from file paths.
- CMO_Peak: Implements the CMO peak detection algorithm.

Functions:
- setGImages: Sets global images for easy access.
- getGImages: Retrieves global images.
- resize: Resizes an image.
- putText: Adds text to an image.
- cv2_img_show: Displays an image using OpenCV.
- putlabel: Adds labels to an image.
- overlay_mask: Overlays a mask on an image.
- draw_bboxes: Draws bounding boxes on an image.
- VideoWriter: Writes video frames to a file.
"""

from .image_loader import ImageLoader as ImageLoader 
from .cmo_peak import CMO_Peak as CMO_Peak

from .g_images import setGImages as setGImages
from .g_images import getGImages as getGImages

from .image_utils import resize as resize
from .image_utils import putText as putText
from .image_utils import cv2_img_show as cv2_img_show
from .image_utils import putlabel as putlabel
from .image_utils import overlay_mask as overlay_mask
from .image_utils import draw_bboxes as draw_bboxes
from .image_utils import VideoWriter as VideoWriter

