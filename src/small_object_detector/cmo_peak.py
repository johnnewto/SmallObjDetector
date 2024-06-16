"""
This module contains the implementation of the CMO_Peak class, which is an object detector module using intensity peaks.
"""

import cv2 as cv2
import numpy as np
from skimage.feature.peak import peak_local_max

from .g_images import *
from .image_utils import BH_op, TH_op, CMO_op, get_tile
from .image_utils import resize, putText, cv2_img_show, putlabel, overlay_mask, draw_bboxes

try:
    # optional use of torch as its big to install
    import torch
    import utils.pytorch_utils as ptu
except:
    pass

import logging
import typing as typ

logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_NUM_TILES = 5

class CMO_Peak():
    """
    Object Detector Module using intensity peaks
    """

    def __init__(self, confidence_threshold=0.5,
                 labels_path=None,
                 expected_peak_max=60,
                 peak_min_distance=10,
                 num_peaks=10,
                 maxpool=12,
                 morph_kernalsize=3,
                 morph_op='BH',
                 track_boxsize=(160, 80),
                 bboxsize=40,
                 draw_bboxes=True,
                 device=None):
        """
        Initializes the CMO_Peak object detector module.

        Args:
            confidence_threshold (float): Confidence threshold for object detection.
            labels_path (str): Path to the labels JSON file.
            expected_peak_max (int): Maximum expected peak value.
            peak_min_distance (int): Minimum distance between peaks.
            num_peaks (int): Number of peaks to detect.
            maxpool (int): Maximum pooling size.
            CMO_kernalsize (int): Kernel size for CMO operation.
            track_boxsize (tuple): Size of the tracking box.
            bboxsize (int): Size of the bounding box.
            draw_bboxes (bool): Whether to draw bounding boxes on the image.
            device (str): Device to use for computation (e.g., "cuda:0" or "cpu").
        """

        object_names = {1: 'plane', 2: 'cloud'}

        self.expected_peak_max = expected_peak_max
        self.peak_min_distance = peak_min_distance
        self.num_peaks = num_peaks
        self.maxpool = maxpool
        self.morph_kernalsize = morph_kernalsize
        self.morph_op = morph_op
        self.bboxsize = bboxsize


        self.object_names = object_names
        self.confidence_threshold = confidence_threshold
        # self.nms_threshold = nms_threshold
        self.height = None
        self.width = None

        np.random.seed(12345)
        if draw_bboxes:
            self.bbox_colors = {key: np.random.randint(0, 255, size=(3,)).tolist() for key in self.object_names.keys()}
            pass


    def set_max_pool(self, maxpool=12):
        """
        Sets the maximum pooling size.

        Args:
            maxpool (int): Maximum pooling size.
        """
        self.maxpool = maxpool
        getGImages().maxpool = self.maxpool
        print(f'Setting maxpool = {self.maxpool}')


    def small_objects(self, morph_op:typ.Union[str, None] = None):
        """ morph_op is 'CMO', 'BH' or 'TH' 
            A Study of Morphological Pre-Processing Approaches for Track-Before-Detect Dim Target Detection
            https://eprints.qut.edu.au/214476/1/16823.pdf
        """

        if morph_op is None:
            morph_op = self.morph_op

        images = getGImages()
        if morph_op == 'CMO':
            images.cmo = CMO_op(images.minpool, (self.morph_kernalsize, self.morph_kernalsize))
        elif morph_op == 'BH':
            images.cmo = BH_op(images.minpool, (self.morph_kernalsize, self.morph_kernalsize))
        elif morph_op == 'BH+filter':
            images.cmo = BH_op(images.minpool, (self.morph_kernalsize, self.morph_kernalsize), filter)
        elif morph_op == 'TH':
            images.cmo = TH_op(images.minpool, (self.morph_kernalsize, self.morph_kernalsize))

        if hasattr(images, 'mask') and images.mask is not None:
            images.cmo[images.mask > 0] = 0   


    def align(self):
        """
        Aligns the image to the last in order to keep tracking centers accurate.

        Returns:
            tuple: Tuple containing the aligned coordinates (row, column).
        """
        try:
            ((sc, sr), _error) = cv2.phaseCorrelate(getGImages().last_minpool_f, getGImages().minpool_f)
            getGImages().last_minpool_f = getGImages().minpool_f
        except:
            getGImages().last_minpool_f = getGImages().minpool_f
            sc, sr = 0, 0

        return round(sr * self.maxpool), round(sc * self.maxpool)

    def get_bb(self, img, pos, threshold=0.5):
        """
        Gets the bounding box from the thresholded contour (usually on a CMO image).

        Args:
            img (numpy.ndarray): Input image.
            pos (tuple): Position of the peak.
            threshold (float): Threshold value.

        Returns:
            list: List containing the bounding box coordinates [x, y, width, height].
        """
        (r1, c1) = pos
        thresh = int(img[r1, c1] * threshold)
        mask = ((img > thresh) * 255).astype('uint8')
        num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_regions > 1:
            idx = labels[r1, c1]
            if idx != 0:
                bb = stats[idx]
                bbwh = [bb[0], bb[1], bb[2], bb[3]]
            else:
                bbwh = None
        else:
            bbwh = None
        if bbwh is None:
            bbwh = [mask.shape[1] // 2, mask.shape[0] // 2, 0, 0]
        return bbwh

    def find_peaks(self):
        """
        Finds the intensity peaks in the image.

        Returns:
            tuple: Tuple containing the peak positions and bounding boxes.
        """
        NUM_PEAKS = 20
        threshold_abs = self.expected_peak_max * self.confidence_threshold
        _pks = peak_local_max(getGImages().cmo,
                              min_distance=self.peak_min_distance,
                              threshold_abs=threshold_abs,
                              num_peaks=NUM_PEAKS)

        self.fullres_cmo_tile_lst = []
        self.fullres_img_tile_lst = []
        self.lowres_cmo_tile_lst = []
        self.lowres_img_tile_lst = []
        self.pks = []
        self.bbwhs = []
        self.pk_vals = []

        # get low res and full res peaks
        # gather all the tiles centered on the peak positions

        # sort the peaks by row
        _pks = sorted(_pks, key=lambda x: x[0])
        # take to first self.num_peaks
        # _pks = _pks[:self.num_peaks]
        # _pks = _pks[:2]

        for (r, c) in _pks:
            # sort value by row, 1.0 at to, and 0.5 at bottom
            rows = getGImages().cmo.shape[0]
            rval = 1.0 - 0.5*(r / rows)

            bs0 = round(self.bboxsize // 2)
            lowres_img = get_tile(getGImages().small_gray, (r - bs0, c - bs0), (self.bboxsize, self.bboxsize))
            lowres_cmo = get_tile(getGImages().cmo, (r - bs0, c - bs0), (self.bboxsize, self.bboxsize))
            self.lowres_img_tile_lst.append(lowres_img)
            self.lowres_cmo_tile_lst.append(lowres_cmo)

            bs = self.bboxsize
            r, c = r * self.maxpool, c * self.maxpool
            img = get_tile(getGImages().full_rgb, (r - bs, c - bs), (bs * 2, bs * 2))
            fullres_cmo = BH_op(img, (self.morph_kernalsize * 2 + 1, self.morph_kernalsize * 2 + 1))
            (_r, _c) = np.unravel_index(fullres_cmo.argmax(), fullres_cmo.shape)
            r, c = r - bs + _r, c - bs + _c
            pk_val_hi_res = fullres_cmo[bs, bs]            # peak value in full res image
            pk_val_lowres = lowres_cmo[bs0, bs0]    # peak value in low res image
            # ave_val = np.mean(fullres_cmo)
            # pk_val = pk_val - ave_val
        
            self.pks.append((r, c, pk_val_lowres, pk_val_hi_res, rval ))
            # self.pk_vals.append(pk_val)

        # sort the peaks by low res peak value
        self.pks = sorted(self.pks, key=lambda x: x[2]*x[4], reverse=True)

        self.pks = self.pks[:self.num_peaks]

        for (r, c, pk_val_hi_res, pk_val_low_res, rv) in self.pks:
            bbwh = [c - bs , r - bs, bs * 2, bs * 2]

            fullres_tile = get_tile(getGImages().full_rgb, (r - bs, c - bs), (bs * 2, bs * 2), copy=True) # copy so any changes to fullres_cmo_tile_lst do not affect the image
            self.fullres_cmo_tile_lst.append(fullres_cmo)  
            self.fullres_img_tile_lst.append(fullres_tile)
            # pk_val = fullres_cmo[bs, bs] 
            # self.pks.append((r, c))
            # self.pk_vals.append(pk_val)

            # convert bbwh to yolo format
            bbwh = [bbwh[0] / self.width, bbwh[1] / self.height, bbwh[2] / self.width, bbwh[3] / self.height]
            self.bbwhs.append(bbwh)
            self.pk_vals.append(pk_val_hi_res)


        logger.info(f'Found {len(self.pks)} peaks {[pk[2] for pk in self.pks]} {[pk[3] for pk in self.pks]} {[pk[4] for pk in self.pks]}')
        # if length tile list < self.num_peakspad with zeros
        while len(self.fullres_img_tile_lst) < self.num_peaks:
            bs2 = self.bboxsize*2
            self.fullres_img_tile_lst.append(np.zeros((bs2, bs2, 3), dtype=np.uint8))
            self.lowres_img_tile_lst.append(np.zeros((bs2, bs2), dtype=np.uint8))
            self.lowres_cmo_tile_lst.append(np.zeros((bs2, bs2), dtype=np.uint8))
            self.pk_vals.append((0, 0, 0, 0, 0))
            self.bbwhs.append([0, 0, 0, 0])

        
        return self.pks, self.bbwhs


    def classify(self, detections, image, pk_vals, pk_gradients, scale, filterClassID):
        """
        Classifies the detections based on peak values and gradients.

        Args:
            detections (list): List of detections.
            image (numpy.ndarray): Input image.
            pk_vals (list): List of peak values.
            pk_gradients (list): List of peak gradients.
            scale (float): Scaling factor.
            filterClassID (list): List of class IDs to filter.

        Returns:
            tuple: Tuple containing the bounding boxes, confidences, class IDs, and full resolution image tiles.
        """
        fullres_img_tile_lst, bboxes, confidences, class_ids = [], [], [], []
        for i, detection in enumerate(detections):
            confidence = pk_vals[i] / self.expected_peak_max
            if pk_gradients[i].min() < 20 and pk_gradients[i].min() < pk_gradients[i].max() // 2:
                class_id = 2  # cloud
            else:
                class_id = 1  # plane or bird
            if confidence > self.confidence_threshold and class_id in filterClassID:
                bs = self.bboxsize // 2
                row, col = int(detection[0] * scale), int(detection[1] * scale)
                bbox = (col - bs, row - bs, col + bs, row + bs)
                bboxes.append(bbox)
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
                fullres_img_tile_lst.append(image[row - bs:row + bs, col - bs:col + bs])

        return bboxes, confidences, class_ids, fullres_img_tile_lst

 
    def classifyNN(self, tile_list):
        """ classify the full res colour tiles """
        batch = ptu.images2batch(tile_list).to(self.device)
        y_pred = self.predictor(batch)
        cat, conf = self.predictor.conf(y_pred)
        class_ids = cat.detach().cpu().numpy().tolist()
        confidences = conf.detach().cpu().numpy().tolist()

        return confidences, class_ids

        # y_pred = self.predictor(batch)
        # cat, conf = self.predictor.conf(y_pred)
        # print(cat)
        # print(conf)



    def detect(self):
        """
        Detect objects in the current image.

        Args:
            inone

        Returns:
            tuple: Tuple containing the following elements:
                - bbwhs List(numpy.ndarray): Bounding boxes with shape (n, 4) containing accurate detected objects with each row as `(xmin, ymin, width, height)`.
                - confidences List(numpy.ndarray): Confidence or detection probabilities if the detected objects with shape (n,).
                - class_ids List(numpy.ndarray): Class_ids or label_ids of detected objects with shape (n, 4)
        """

        self.height, self.width = getGImages().full_rgb.shape[:2]
        pks, bbwhs = self.find_peaks()

        confidences = [pk[2] / 255.0 for pk in pks]  # peak value in low res image
        class_ids = [0] * len(self.fullres_img_tile_lst)

        # already sorted by low res peak value

                # try:
                #     # sort lists by confidences (peak values)
                #     zipped_lists = list(zip( bbwhs, confidences, class_ids))
                #     # Sort the zipped lists by confidences (the third element in each tuple)
                #     sorted_lists = sorted(zipped_lists, key=lambda x: x[2], reverse=True)
                #     # Unzip the sorted list back into individual lists
                #     bbwhs, confidences, class_ids = zip(*sorted_lists)
                # except Exception as e:
                #     logger.warning(e)

        return  bbwhs, confidences, class_ids


    def old_draw_bboxes(self, image, bboxes, confidences, class_ids, display_scale=None, text=True, thickness=6, alpha:typ.Union[float, None]=0.3):
        """
        Draw the bounding boxes about detected objects in the image.

        Args:
            image (numpy.ndarray): Image or video frame.
            bboxes (numpy.ndarray): Bounding boxes pixel coordinates as (xmin, ymin, width, height)
            confidences (numpy.ndarray): Detection confidence or detection probability.
            class_ids (numpy.ndarray): Array containing class ids (aka label ids) of each detected object.
            display_scale: ratio of bbbox coordinates to the actual display coords

        Returns:
            numpy.ndarray: image with the bounding boxes drawn on it.
        """
        if confidences is None:
            confidences = [1.0 for bb in bboxes]
        if class_ids is None:
            class_ids = [i for i, bb in enumerate(bboxes)]

        # support for alpha blending overlay
        overlay = image.copy() if alpha is not None else image    
        count = 0
        for bb, conf, cid in zip(bboxes, confidences, class_ids):
            # clr = [int(c) for c in self.bbox_colors[cid]]
            if count < 5:
                clr = (255, 0, 0)
            elif  count < 10:
                clr = (0,255,0)
            else:
                clr = (0, 0, 255)


            if display_scale is not None:
                for i in range(len(bb)):
                    bb[i] = int(bb[i] * display_scale)
                # bb[0], bb[1] = int(bb[0]*display_scale), int(bb[1]*display_scale)
            
            
            cv2.rectangle(overlay, (bb[0], bb[1] ), (bb[0] + bb[2], bb[1] + bb[3]), clr, thickness)
            if text:
                _font_size = 0.8
                _thickness = 2
                label = f"{self.object_names[cid]} : {conf:.2f}"
                label = f"{count}"
                (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _font_size, _thickness)
                y_label = max(bb[1], label_height)
                cv2.rectangle(overlay, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
                             (255, 255, 255), cv2.FILLED)
                cv2.putText(overlay, label, (bb[0], y_label+5), cv2.FONT_HERSHEY_SIMPLEX, _font_size, clr, _thickness)
                count += 1
        
        if alpha is not None:
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image

    def display_results(self, image, alpha=0.3):

        if getGImages().mask is not None:
            disp_image = overlay_mask(image, getGImages().mask, alpha=alpha)
        else:
            disp_image = image
        disp_image = draw_bboxes(disp_image, self.bbwhs, self.pks, text=True, thickness=8, alpha=alpha)
        for count, tile in enumerate (self.fullres_img_tile_lst):
            # put label count in left top corner
            clr = (255, 0, 0) if count < 5 else (0,255,0) if count < 10 else (0, 0, 255)  # in order red, green, blue
            # cv2.putText(tile, str(count), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)
            putlabel(tile, f'{count}', (0,10), fontScale=0.4, color=clr, thickness=1)
        try:

            tiles = []
            if len(self.fullres_img_tile_lst) < MAX_NUM_TILES:
                tiles = [np.zeros_like(tile)] * (MAX_NUM_TILES - len(self.fullres_img_tile_lst))
            tile_img = np.hstack(self.fullres_img_tile_lst+tiles)


            tile_img = resize(tile_img, width=image.shape[1], inter=cv2.INTER_NEAREST)
            disp_image = np.vstack([tile_img, disp_image])
        except Exception as e:
            logger.error(e)
                                                    
        return disp_image
        # for bb, conf, cid in zip(bboxes, confidences, class_ids):
        #     clr = [int(c) for c in self.bbox_colors[cid]]
        #     cv.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 1)
        #     label = "{}:{:.4f}".format(self.object_names[cid], conf)
        #     (label_width, label_height), baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #     y_label = max(bb[1], label_height)
        #     cv.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
        #                  (255, 255, 255), cv.FILLED)
        #     cv.putText(image, label, (bb[0], y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
        # return image