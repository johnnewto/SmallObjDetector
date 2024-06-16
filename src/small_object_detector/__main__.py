"""
This file contains the main code for the MauiTracker application.
The code includes classes and functions for processing and analyzing video frames.

Classes:
- Viewer: Represents the class for processing and analyzing video frames.

Functions:
- __call__: Runs the video processing and analysis.


Version:
--------

- small-object-detector v0.0.2
"""


import logging
import time
import cv2
from pathlib import Path
from small_object_detector import ImageLoader
from small_object_detector import CMO_Peak
from small_object_detector import setGImages, getGImages
from small_object_detector import resize, putText, cv2_img_show, VideoWriter
# from utils.show_images import  putText, cv2_img_show

# import typing as typ

logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Viewer:
    def __init__(self, _loader, detecter, tracker, display_width=2000, record=False, record_name=None, frame_rate=5, path='', qgc=None):
        self.loader = _loader
        self.detecter: CMO_Peak = detecter
        self.tracker = tracker
        self.display_width = display_width
        self.record = record
        self.record_name = record_name
        self.frame_rate = frame_rate
        self.path = path

        self.detecter.set_max_pool(12)
        self.heading_angle = 0.0



    def __call__(self, wait_timeout=10):
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
        # cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if self.record:
            if self.record_name:
                filename = f"{self.record_name}.mp4"
            else:
                filename = f"{self.path}-{self.detecter.morph_op}.mp4"
            video = VideoWriter(filename, fps=self.frame_rate)
            print(f"Recording to {filename}")


        first_run = True
  
 
        k = -1
        for (image, filename), frameNum, grabbed in iter(self.loader):

            if grabbed or first_run:
                first_run = False
                # print(f"frame {frameNum} : {filename}  {grabbed}")
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
                setGImages(image)
                # getGImages().mask_sky()
                # cv2_img_show('find_sky_2-mask', getGImages().mask, flags=cv2.WINDOW_NORMAL)

        

                self.detecter.small_objects()
                cv2_img_show(f'{self.detecter.morph_op}', getGImages().cmo, flags=cv2.WINDOW_NORMAL, normalise=True)    # f'{self.detecter.morph_op}'

                self.detecter.detect()
                height, width, _ = image.shape
                putText(image, f'Frame# = {frameNum}, {filename}', row=height-80, fontScale=0.5)
                disp_image = self.detecter.display_results(image)


                if self.record:
                    img = resize(disp_image, width=3000)  # not sure why this is needed to stop black screen video
                    # for i in range(2):  # to allow easier pause and frame seeking in video play
                    video.add(img)

            cv2_img_show(WindowName, disp_image, win_width=self.display_width)
            # try:
            #     cv2_img_show('fullres_tiles', vstack(
            #         [np.hstack(self.model.fullres_img_tile_lst), np.hstack(self.model.fullres_cmo_tile_lst)]),
            #                  height=200)

            #     cv2_img_show('lowres_tiles', vstack(
            #         [np.hstack(self.model.lowres_img_tile_lst), np.hstack(self.model.lowres_cmo_tile_lst)]),
            #                  height=200)
            # except Exception as e:
            #     logger.error(e)
                       
            k = cv2.waitKey(wait_timeout)
            if k == ord('q') or k == 27:
                break    

            if k == ord('q') or k == 27:
                self.do_run = False
                break
            if k == ord(' '):
                wait_timeout = 0
            if k == ord('g'):
                wait_timeout = 1
            if k == ord('/'):
                # change direction
                wait_timeout = 0
                self.loader._direction_fwd = not self.loader._direction_fwd

            if k == ord('d'):  # increment 10 frames
                print("increment 10 frames")
                self.loader.frame_num = min(self.loader.frame_num + 10, self.loader.num_frames - 1)
                print(f"frame_num = {self.loader.frame_num}")
            if k == ord('a'):  # decrement 10 frames
                print("decrement 10 frames")
                self.loader.frame_num = max(self.loader.frame_num - 10, 0)

            if k == ord('s'):  # increment 100 frames
                print("increment 100 frames")
                self.loader.frame_num = min(self.loader.frame_num + 100, self.loader.num_frames - 1)
                print(f"frame_num = {self.loader.frame_num}")
            if k == ord('w'):  # decrement 100 frames
                print("decrement 100 frames")
                self.loader.frame_num = max(self.loader.frame_num - 100, 0)

        if self.record:
            video.close()
 

        cv2.destroyAllWindows()
        self.loader.close()

        time.sleep(0.5)

    @staticmethod
    def help():
        print(" Tracking of small objects in video frames")
        print("   keys: q     : quit")
        print("         space : pause")
        print("         g     : go")
        print("         .     : change direction")
        print("         d     : increment 10 frames")
        print("         a     : decrement 10 frames")
        print("         s     : increment 100 frames")
        print("         w     : decrement 100 frames")
        print("     ")

def main():
    import argparse
    """

    """

    _tracker = None

    parser = argparse.ArgumentParser(description=Viewer.help())
    parser.add_argument('-r', '--record', action='store_true', help='Enable recording, using default "dir path" name if not defined with -n NAME', default=False)
    parser.add_argument('-n', '--name', type=str, help='record name', default=None)
    parser.add_argument('-f', '--framerate', type=int, help='recording framerate', default=5)
    parser.add_argument('-d', '--dir', type=str, help='directory to view', default=None)
    parser.add_argument('-s', '--start', type=int, help='start frame', default=None)   
    parser.add_argument('-e', '--end', type=int, help='end frame', default=None)
    parser.add_argument('-t', '--tracks', type=int, help='number of objects to track', default=5)

    # add Viewer.help() to parser help

    args = parser.parse_args()

    RECORD = args.record
    RECORD_NAME = args.name 
    FRAME_RATE = args.framerate
    START_FRAME = args.start
    END_FRAME = args.end
    NUM_PEAKS = args.tracks
    RECORD = True
    # RECORD_NAME = 'temp'

    # home = str(Path.home())
    detecter = CMO_Peak(confidence_threshold=0.1,
                      labels_path='data/imagenet_class_index.json',
                      # labels_path='/media/jn/0c013c4e-2b1c-491e-8fd8-459de5a36fd8/home/jn/data/imagenet_class_index.json',
                      expected_peak_max=60,
                      peak_min_distance=5,
                      num_peaks=NUM_PEAKS,
                      maxpool=12,
                      morph_kernalsize=3,
                      morph_op='BH+filter',
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
    if args.dir:
        path = args.dir
    else:
        home = str(Path.home())
        # if data path exists use it
        path = home + '/data/maui-data/Karioitahi_09Feb2022/132MSDCF-28mm-f4/'
        # path = home + '/data/maui-data/karioitahi_13Aug2022/SonyA7C/105MSDCF'

        # if not os.path.exists(path):
        #     print(f"Path {path} does not exist, using local path")
        #     path = "data/Karioitahi_09Feb2022/132MSDCF-28mm-f4"

    cv2.destroyAllWindows()

    loader = ImageLoader(path, names=('*.jpg','*.JPG'), mode='RGB', cvtgray=False, start_frame=START_FRAME, end_frame=END_FRAME)
  
    view = Viewer(loader, detecter, _tracker, display_width=1000, record=RECORD, frame_rate=FRAME_RATE, path=path)

    view(wait_timeout=0)


if __name__ == '__main__':

    main()
