"""
This file contains the main code for the MauiTracker application.
The code includes classes and functions for processing and analyzing video frames.

Classes:
- Main: Represents the main class for processing and analyzing video frames.

Functions:
- run: Runs the video processing and analysis.

"""

import logging
import os
import time
import cv2
from pathlib import Path
import numpy as np
import utils.image_loader as il
from utils.cmo_peak import CMO_Peak
from utils.g_images import setGImages, getGImages
from utils.image_utils import resize, putText, cv2_img_show, putlabel, overlay_mask, draw_bboxes
# from utils.show_images import  putText, cv2_img_show

# import typing as typ

logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)





class Main:
    def __init__(self, _loader, model, tracker, display_width=2000, record=False, path='', qgc=None):
        self.loader = _loader
        self.model: CMO_Peak = model
        self.tracker = tracker
        self.display_width = display_width
        self.record = record
        self.do_run = True
        self.path = path

        self.model.set_max_pool(12)
        self.heading_angle = 0.0



    def run(self, wait_timeout=10):
        """
            Run the main tracking loop.

            Args:
                wait_timeout (int, optional): The wait timeout in milliseconds. Defaults to 10.
                heading_angle (int, optional): The heading angle. Defaults to 0.
                stop_frame (int, optional): The frame number to stop at. Defaults to None.
            """
        WindowName = "Main View"
        cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)

        # These two lines will force your "Main View" window to be on top with focus.
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if self.record:
            video = VideoWriter(self.path + '.mov', fps=5.0)
            print(f"Recording to {self.path}.mov")


        first_run = True
        while self.do_run:
            k = -1
            for (image, filename), frameNum, grabbed in iter(self.loader):

                if grabbed or first_run:
                    first_run = False
                    print(f"frame {frameNum} : {filename}  {grabbed}")
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
                    setGImages(image)
                    getGImages().mask_sky()
                    getGImages().small_objects()

                    self.model.detect()
                    disp_image = self.display_results(image)
 
                    putText(disp_image, f'Frame# = {frameNum}, {filename}', row=170, fontScale=0.5)

                    if self.record:
                        img = resize(disp_image, width=3000)  # not sure why this is needed to stop black screen video
                        for i in range(2):  # to allow easier pause and frame seeking in video play
                            video.add(img)

                cv2_img_show(WindowName, disp_image)
                # try:
                #     cv2_img_show('fullres_tiles', vstack(
                #         [np.hstack(self.model.fullres_img_tile_lst), np.hstack(self.model.fullres_cmo_tile_lst)]),
                #                  height=200)

                #     cv2_img_show('lowres_tiles', vstack(
                #         [np.hstack(self.model.lowres_img_tile_lst), np.hstack(self.model.lowres_cmo_tile_lst)]),
                #                  height=200)
                # except Exception as e:
                #     logger.error(e)

                
    
                if k == ord('q') or k == 27:
                    self.do_run = False
                    break
                if k == ord(' '):
                    wait_timeout = 0
                if k == ord('g'):
                    wait_timeout = 1
                if k == ord('d'):
                    # change direction
                    wait_timeout = 0
                    self.loader._direction_fwd = not self.loader._direction_fwd

                k = cv2.waitKey(wait_timeout)

            # self.loader.direction_fwd = not self.loader.direction_fwd
            wait_timeout = 0

            k = cv2.waitKey(wait_timeout)
            if k == ord('q') or k == 27:
                break

        if self.record:
            video.close()
 

        cv2.destroyAllWindows()
        self.loader.close()

        time.sleep(0.5)

        # """
        # Put text on the image with a black background.

        # Args:
        #     img (numpy.ndarray): The image.
        #     text (str): The text to put on the image.
        #     position (tuple): The position where the text should be put.
        #     fontFace (int): The font type. Default is cv2.FONT_HERSHEY_SIMPLEX.
        #     fontScale (float): Font scale. Default is 1.
        #     color (tuple): Font color. Default is white.
        #     thickness (int): Thickness of the lines used to draw a text. Default is 2.

        # Returns:
        #     numpy.ndarray: The image with the text.
        # """
        # # Calculate the width and height of the text
        # (text_width, text_height), baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
        # # Determine the y-coordinate for the text
        # y_text = max(position[1], text_height)
        # # Draw a filled rectangle on the image at the location where the text will be placed
        # cv2.rectangle(img, (position[0], y_text - text_height), (position[0] + text_width, y_text + baseLine), (0, 0, 0), cv2.FILLED)
        # # Draw the text on the image at the specified location
        # cv2.putText(img, text, (position[0], y_text+text_height//4), fontFace, fontScale, color, thickness, cv2.LINE_AA)

        # return img

    def display_results(self, image, alpha=0.3):

        disp_image = overlay_mask(image, getGImages().mask, alpha=alpha)
        disp_image = draw_bboxes(disp_image, self.model.bbwhs, self.model.pks, text=True, thickness=8, alpha=alpha)
        for count, tile in enumerate (self.model.fullres_img_tile_lst):
            # put label count in left top corner
            clr = (255, 0, 0) if count < 5 else (0,255,0) if count < 10 else (0, 0, 255)  # in order red, green, blue
            # cv2.putText(tile, str(count), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)
            putlabel(tile, f'{count}', (0,10), fontScale=0.4, color=clr, thickness=1)
        try:
            tile_img = np.hstack(self.model.fullres_img_tile_lst)
            tile_img = resize(tile_img, width=self.display_width, inter=cv2.INTER_NEAREST)
            disp_image = np.vstack([tile_img, disp_image])
        except Exception as e:
            logger.error(e)
                                                       
        return disp_image


if __name__ == '__main__':
    import argparse
    """

    """

    # RECORD = False
    STOP_FRAME = None

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.2
    DRAW_BOUNDING_BOXES = True

    
    _tracker = None

    parser = argparse.ArgumentParser(description="Tracking of small objects in video frames")
    parser.add_argument('-r', '--record', action='store_true', help='Enable recording', default=False)
    args = parser.parse_args()

    RECORD = args.record

    # home = str(Path.home())
    _model = CMO_Peak(confidence_threshold=0.1,
                      labels_path='data/imagenet_class_index.json',
                      # labels_path='/media/jn/0c013c4e-2b1c-491e-8fd8-459de5a36fd8/home/jn/data/imagenet_class_index.json',
                      expected_peak_max=60,
                      peak_min_distance=5,
                      num_peaks=10,
                      maxpool=12,
                      CMO_kernalsize=3,
                      track_boxsize=(80, 160),
                      bboxsize=40,
                      draw_bboxes=True,
                      device=None, )

    # # gen_cmo = GenCMO(shape=(600,1000), dt=0.1, n=5)
    # (rows, cols) = (2000, 3000)
    # center = (2750, 4350)
    # (_r, _c) = (center[0] - rows // 2, center[1] - cols // 2)
    # crop = [_r, _r + rows, _c, _c + cols]
    # # crop = None
    home = str(Path.home())

    # if data path exists use it
    path = home + '/data/maui-data/Karioitahi_09Feb2022/132MSDCF-28mm-f4' 
    # path = home + '/data/maui-data/Karioitahi_09Feb2022/136MSDCF'
    # path = home + '/data/maui-data/Karioitahi_15Jan2022/125MSDCF-landing'
    # path = home + '/data/maui-data/Tairua_15Jan2022/116MSDCF'
    # path = home + '/data/maui-data/Tairua_15Jan2022/109MSDCF'
    # path = home + '/data/maui-data/karioitahi_13Aug2022/SonyA7C/103MSDCF'
    path += '-use-local-path'
    if not os.path.exists(path):
        print(f"Path {path} does not exist, using local path")
        path = "data/Karioitahi_09Feb2022/132MSDCF-28mm-f4"


    loader = il.ImageLoader(path + '/*.JPG', mode='RGB', cvtgray=False, start_frame=0)
  
    main = Main(loader, _model, _tracker, display_width=6000, record=RECORD, path=path)

    main.run(wait_timeout=0)


