# __all__ = ['setCurrentImages',  'getCurrentImages', 'g_images', 'Images',
__all__ = ['scale_image', 'resize', 'crop_image', 'ImageLoader', 'CMO_op', 'BH_op', 'TH_op' ,'make_tile_list', 'old_make_tile_list', 'crop_idx',
           'tile_images', 'get_project_root', 'max_pool', 'min_pool','count_pool', 'norm_uint8',
           'get_tile']
# , 'find_sky_1', 'find_sky_2' , 'near_mask', 'mask_horizon_1', 'mask_horizon_2',]

import cv2 as cv2
import numpy as np
from imutils import resize


from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

import time, sys
import typing as typ
import logging
from pathlib import Path

if sys.platform == "win32":
    # On Windows, the best timer is time.clock()
    default_timer = time.perf_counter
else:
    # On most other platforms the best timer is time.time()
    default_timer = time.time



logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def get_project_root() -> Path:
    return Path(__file__).parent.parent

def static(**kw):
    def decorator(f):
        f.__dict__.update(kw)
        return f
    return decorator


def timeval(func):
    def wrapper(*arg, **kw):
        t1 = default_timer()
        # res = func(*arg, **kw)
        t2 = default_timer()
        print (f'{t2 - t1} {func.__name__}')
    return wrapper

def crop_idx(row, col, bs, image_shape):
    """ crop the indexes so that bbox won't be to cropped at image boundaries"""
    row = max(row, bs)
    row = min(row, image_shape[0]-1)
    col = max(col, bs)
    col = min(col, image_shape[1]-1)
    return row, col

def norm_uint8(img):
    return cv2.normalize(img, img, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def BH_op(img, kernelSize, filter=False):
    """bottom-hat transformation Filtering Approach
    A Study of Morphological Pre-Processing Approaches for Track-Before-Detect Dim Target Detection
    https://eprints.qut.edu.au/214476/1/16823.pdf
    """
    if img.ndim == 2:
        obs = img
    else:
        # observation is gray image
        obs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _cmo = closing(obs, selem) - opening(obs, selem)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    # blackhat = cv2.morphologyEx(obs, cv2.MORPH_CLOSE, kernel) - obs
    blackhat = cv2.morphologyEx(obs, cv2.MORPH_BLACKHAT, kernel)
    if filter:
        # kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.int8)
        kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.int8)
        # kernel = np.array([[0,0,-1,0,0],[0,0,-1,0,0], [-1,-1,8,-1,-1], [0,0,-1,0,0], [0,0,-1,0,0]], np.int8)        
        blackhat = cv2.filter2D(blackhat,-1,kernel)   
        # blackhat = cv2.filter2D(blackhat,-1,kernel)  
        # blackhat = cv2.filter2D(blackhat,-1,kernel) 
        # kernel = cv2.getGaussianKernel(5, 1, cv2.CV_32F)
        # blackhat = cv2.sepFilter2D(blackhat,-1,kernel,kernel)

    return blackhat

def TH_op(img, kernelSize, shape=cv2.MORPH_RECT):
    """top-hat transformation Filtering Approach
    A Study of Morphological Pre-Processing Approaches for Track-Before-Detect Dim Target Detection
    https://eprints.qut.edu.au/214476/1/16823.pdf
    """
    if img.ndim == 2:
        obs = img
    else:
        # observation is gray image
        obs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _cmo = closing(obs, selem) - opening(obs, selem)
    kernel = cv2.getStructuringElement(shape, kernelSize)
    # tophat = obs - cv2.morphologyEx(obs, cv2.MORPH_OPEN, kernel)
    tophat = cv2.morphologyEx(obs, cv2.MORPH_TOPHAT, kernel)
    return tophat

def CMO_op(img, kernelSize):
    """Close-Minus-Open Filtering Approach
    A Study of Morphological Pre-Processing Approaches for Track-Before-Detect Dim Target Detection
    https://eprints.qut.edu.au/214476/1/16823.pdf
    """
    if img.ndim == 2:
        obs = img
    else:
        # observation is gray image
        obs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _cmo = closing(obs, selem) - opening(obs, selem)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    _cmo = cv2.morphologyEx(obs, cv2.MORPH_CLOSE, kernel) - cv2.morphologyEx(obs, cv2.MORPH_OPEN, kernel)
    return _cmo

# def DOT_op(img, kernelSize):
#     """Close-Minus-Open Filtering Approach
#     A Study of Morphological Pre-Processing Approaches for Track-Before-Detect Dim Target Detection
#     https://eprints.qut.edu.au/214476/1/16823.pdf
#     """
#     if img.ndim == 2:
#         obs = img
#     else:
#         # observation is gray image
#         obs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # _cmo = closing(obs, selem) - opening(obs, selem)
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
#     kernel =  np.array([[0, 0, 0],
#                         [0, 1, 0],
#                         [0, 0, 0]], dtype='uint8')
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     opening = cv2.morphologyEx(obs, cv2.MORPH_OPEN, kernel, iterations=1)
#     # _cmo = cv2.morphologyEx(obs, cv2.MORPH_CLOSE, kernel) - cv2.morphologyEx(obs, cv2.MORPH_OPEN, kernel)
#     return opening


def old_max_pool(mat, K, L):
    M, N = mat.shape
    MK = M // K
    NL = N // L
    return mat[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3))

def old_min_pool(mat, K, L):
    M, N = mat.shape
    MK = M // K
    NL = N // L
    return mat[:MK * K, :NL * L].reshape(MK, K, NL, L).min(axis=(1, 3))

def max_pool(mat, K, L):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (K, L))
    return cv2.morphologyEx(mat, cv2.MORPH_DILATE, kernel)[::K, ::L]

def min_pool(mat, K, L):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (K, L))
    return cv2.morphologyEx(mat, cv2.MORPH_ERODE, kernel)[::K, ::L]

def count_pool(mat, K, L):
    M, N = mat.shape
    MK = M // K
    NL = N // L
    return np.count_nonzero(mat[:MK * K, :NL * L].reshape(MK, K, NL, L), axis=(1, 3))

def get_tile(img:np.ndarray, pos, tile_shape, copy=False):
    '''
    get a `tile` with values from `img` at `pos`,
    while accounting for the tile being off the edge of `img`.
    *Note:* negative values in `pos` are interpreted as-is, not as "from the end".
    '''

    img_shape, pos, tile_shape = np.array(img.shape[:2]), np.array(pos), np.array(tile_shape[:2])
    end = pos + tile_shape[:2]        # Calculate img slice positions

    if np.all(pos >= 0) and np.all(img_shape-end >= 0):
        tile = img[pos[0]:end[0], pos[1]:end[1]]

    else:
        if img.ndim == 3:
            if img.shape[2] == 3:
                tile = np.zeros((*tile_shape, 3), 'uint8')
            elif img.shape[2] == 4:
                tile = np.zeros((*tile_shape, 4), 'uint8')
            else:
                print("error")  # FIXME add error handling
        else:
            tile = np.zeros(tile_shape, 'uint8')
        # Calculate tile slice positions
        crop_low = np.clip(0 - pos, a_min=0, a_max=tile_shape)
        crop_high = tile_shape - np.clip(end - img_shape, a_min=0, a_max=tile_shape)
        crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
        # Calculate img slice positions
        pos = np.clip(pos, a_min=0, a_max=img_shape)
        end = np.clip(end, a_min=0, a_max=img_shape)
        img_slices = (slice(low, high) for low, high in zip(pos, end))
        try:
            tile[tuple(crop_slices)] = img[tuple(img_slices)]
        except Exception as e:
            print(e)  # FIXME add error handling

    if not np.all(tile.shape[:2] == tile_shape) :
        print("error")
    try:
        assert np.all(tile.shape[:2] == tile_shape)
    except Exception as e:
        print(e)

    return tile.copy() if copy else tile


def scale_image(img, scale=0.5):
    if scale != 1.0:
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        return img

def crop_image(img, crop=None):
    if crop is not None:
        return img[crop[0]:crop[1], crop[2]:crop[3]]
    else:
        return img

def  error_image(text='Error', width=600, height=400):
    return puttext(np.zeros((height,width),dtype=np.uint8), text)

def puttext(img, text, pos=None, fontscale=1.0, textcolor=(255,255,255), backcolor=(0,0,0)):
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 1)[0]
    if pos is None:
        pos = (img.shape[1]//2-text_width//2, img.shape[0]//2+text_height//2)

    rs_point = (pos[0]-2, pos[1]+2)
    re_point = (pos[0]+text_width+4, pos[1]-text_height-2)
    img = cv2.rectangle(img, rs_point, re_point, backcolor, -1)
    return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fontscale, textcolor, 1, cv2.LINE_AA)

def hline(img, row, color, thickness=2):
    """ plot a horizontal line, move it up or down so the show full thickness """
    row = thickness//2 if row < thickness else row
    row = img.shape[0]-thickness//2 if row > img.shape[0]-thickness//2 else row
    cv2.line(img, (0, row), (img.shape[1], row), color, thickness)

def vline(img, col, color, thickness=2):
    """ plot a vertical line, move it left or right so the show full thickness """
    col = thickness//2 if col < thickness else col
    col = img.shape[1]-thickness//2 if col > img.shape[1]-thickness//2 else col
    cv2.line(img, (col, 0), (col, img.shape[0]), color, thickness)

def old_make_tile_list(image: np.ndarray, tracks, tile_size=80,  display_scale=1, which=None):
    """
    Trawl through all the tracks to make a list of tiles
    - track is a list of Tuples of 10 elements representing (frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)
    - which is an array of bools for which to include
    - Returns list of tuple(numpy, labelID, classID, pos(r,c), frameID)
    """
    #
   
    tile_list = []
    numTracks = len(tracks)
    if which is None:
        which = np.ones(numTracks, dtype=bool)
    try:
        # try pyimagesearch tracker
        for (objectID, trk) in tracks.items():
            # (frame_ID, labelID, col, row, w, h, confidence, classID, *z) = trk
            col, row = trk.centroid[0], trk.centroid[1]
            labelID = objectID
            w, h = tile_size//2,tile_size//2
            row = image.shape[0] - h if row >= image.shape[0] - h else row
            col = image.shape[1] - w if col >= image.shape[1] - w else col
            row = h if row < h else row
            col = w if col < w else col
            img = image[row-h:row+h, col-w:col+w].copy()
            if img.shape[:2] == (h*2, w*2):
                tile_list.append((img, labelID, trk.confidence, 1, (row, col),1))  # tuple with (numpy, labelID, defect pos. frameID)
            else:
                logger.error(f'tile shape is wrong {img.shape}')

    except:
        for trk in tracks:
            row = int((trk[1]+trk[3])/2)
            col = int((trk[0]+trk[2])/2)
            labelID = int(trk[4])
            score = trk[5]
            img = get_tile(image, (row, col), (40,40))
            tile_list.append((img, labelID, score, 1, (row, col), 1))  # tuple with (numpy, labelID, defect pos. frameID)

        # for i, trk in enumerate(tracks):
        #     (frame_ID, labelID, col, row, w, h, confidence, classID, *z) = trk
        #     (col, row, h, w) = [round(v / scale) for v in (col, row, h, w)]
        #     if which[i]:
        #         row = image.shape[0] - h if row >= image.shape[0]-h else row
        #         col = image.shape[1] - w if col >= image.shape[1]-w else col
        #         img = image[row:row + h, col:col + w].copy()
        #         if  img.shape[:2] == (h,w):
        #             tile_list.append((img, labelID, confidence, classID, (row, col), frame_ID)) # tuple with (numpy, labelID, defect pos. frameID)
        #         else:
        #             logger.error(f'tile shape is wrong {img.shape}')


    return tile_list

def make_tile_list(tile_list, tracks, which=None):
    """
    Trawl through all the tracks to make a list of tiles
    - track is a list of Tuples of 10 elements representing (frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)
    - which is an array of bools for which to include
    - Returns list of tuple(numpy, labelID, classID, pos(r,c), frameID)
    """
    #
    # scale = 1
    tile_tuple_list = []
    assert len(tracks) == len(tile_list)
    numTracks = len(tracks)
    if which is None:
        which = np.ones(numTracks, dtype=np.bool)
    for i, trk in enumerate(tracks):
        (frame_ID, labelID, col, row, _, _, confidence, classID, *_) = trk
        # (col, row, h, w) = [round(v / scale) for v in (col, row, h, w)]
        if which[i]:
            # row = image.shape[0] - h if row >= image.shape[0]-h else row
            # col = image.shape[1] - w if col >= image.shape[1]-w else col
            # img = image[row:row + h, col:col + w].copy()
            img = tile_list[i]
            tile_tuple_list.append((img, labelID, confidence, classID, (row, col), frame_ID)) # tuple with (numpy, labelID, defect pos. frameID)

    return tile_tuple_list


def tile_images(img_list, horz=True, label=True, fontscale=0.4, tile_size=100, txtpos=(5,15), max_tiles=10, colors=None):
    """ inglist is a list of tuple(img, labelID, confidence, classID, (row, col), frame_ID)"""
    # imgList = sorted(img_list, key=lambda x: x[3])
    imgs_l = []
    cnt=0
    lst = sorted(img_list, key=lambda id: id[1])
    for i,tpl in enumerate(lst):
        if i == max_tiles:
            break
        (img, labelID, confidence, classID) = tpl[:4]
        if tile_size is not None:
            img = resize(img, width=tile_size)
        if colors is not None:
            clr = colors[classID]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            clr = None

        # img[0,:] = 0; img[-1,:] = 0; img[:,0] = 0; img[:,-1] = 0
        if label:
            puttext(img, f'{labelID}: {classID}: {confidence:.2f}', pos=txtpos, fontscale=fontscale, backcolor=clr)
        imgs_l.append(img)
        cnt += 1

    for i in range(cnt,max_tiles,1):
        img = np.full((tile_size, tile_size), 127, dtype=np.uint8)
        # img[0,:] = 0; img[-1,:] = 0; img[:,0] = 0; img[:,-1] = 0
        if colors is not None:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imgs_l.append(img)


    if horz:
        return np.hstack(imgs_l)
    else:
        return np.vstack(imgs_l)


def old_tile_images(imgList, rows=1, label=True, tile_width=None, tile_height=None, txtpos=(5,30), textcolor=(255,255,0), linecolor=(255,255,0), thickness=2, scale=1):
    """
    tile image list together to display in a single window, cols: the num of img in a row
    imgList consists of tuples  (numpy, labelID, [defect pos], [frameID])  [ ] = optional
    """
    if len(imgList) == 0 :
        return error_image('Empty List')
    if type(imgList[0]) != tuple:
        return error_image('Not list of tuples')

    rows = 1 if rows < 1 else rows
    totalImages = len(imgList)
    cols = totalImages // rows if totalImages // rows * rows == totalImages else totalImages // rows + 1
    (height, width, *_) = imgList[0][0].shape

    img = np.zeros((height*rows, width*cols), np.uint8)
    for x in range(cols):
        for y in range(rows):
            idx = x * rows + y
            rpos = y*height
            if idx < len(imgList):
                cpos = x*width
                img[rpos:rpos+height, cpos:cpos+width] = imgList[idx][0]

    if scale != 1:
        img = cv2.resize(img, (0, 0), None, scale, scale)
        width = int(width * scale)
        height = int(height * scale)
    if label:
        for r in range(0, img.shape[0]+1, height):  # plus 1 to include bottom line
            hline(img, r, linecolor, thickness)
        for c in range(0, img.shape[1]+1, width):
            vline(img, c, linecolor, thickness)

        for x in range(cols):
            for y in range(rows):
                if idx < len(imgList):
                    idx = x * rows + y
                    rpos = y * height
                    cpos = x * width
                    if len(imgList[idx]) > 1:
                        img = puttext(img, f'{imgList[idx][1]}', pos=(cpos+txtpos[0], rpos+txtpos[1]), textcolor=textcolor)
                    if len(imgList[idx])>3:
                        img = puttext(img, f'{imgList[idx][3]}', pos=(cpos+txtpos[0], rpos+txtpos[1]+height-40), textcolor=textcolor)
                    if len(imgList[idx]) > 2:
                        rpos_t = rpos + int(imgList[idx][2][0] * scale)
                        cpos_t = cpos + width//2
                        cv2.line(img, (cpos, rpos_t), (cpos + 10, rpos_t), linecolor, 2)
                        cv2.line(img, (cpos + width, rpos_t), (cpos + width - 10, rpos_t), linecolor, 2)
                        cv2.line(img, (cpos_t, rpos), (cpos_t, rpos + 10), linecolor, 2)
                        cv2.line(img, (cpos_t, rpos + height), (cpos_t, rpos + height - 10), linecolor, 2)

                    # if crosshair:


    return img


def load_mp4(fn):
    """ load mp4 file into np array"""
    cap = cv2.VideoCapture(fn)

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = frame[:,:]
            frames.append(img)
            print('.', end='')
        else:
            break

    cap.release()
    print('')
    print('Convert to numpy')
    return np.asarray(frames)

class ImageLoader:
    """Load a sequence of images"""
    def __init__(self, fn, crop=None, scale=1.0, color='Gray'):
        self.crop = crop
        self.scale = scale
        self.color = color
        self._stack = np.load(fn)
        self.frame_num = -1
        self.shape = self._stack.shape
        self.direction_fwd = True
        self.restep = False

    def set_crop(self, center, width=0.5, height=None):
        (_, rows, cols) = self._stack.shape
        if height is None:
            height = width
        rows, cols = int(rows * height), int(cols * width)
        (_r, _c) = (center[0] - rows // 2, center[1] - cols // 2)
        self.crop = [_r, _r + rows, _c, _c + cols]
        self.shape = (self.shape[0], rows, cols)

    def set_frame_num(self, frame_num):
        assert frame_num < self._stack.shape[0], f"frame_num ({frame_num}) > number of images ({self._stack.shape[0]}) in stack"
        self.frame_num = frame_num

    def __iter__(self):
        return self

    def __next__(self):
        # step = 1 if self.direction_fwd else -1
        if not self.restep:
            self.frame_num += 1 if self.direction_fwd else -1
        self.restep = False

        if self.frame_num >= 0 and self.frame_num < self._stack.shape[0]:
            img = self._stack[self.frame_num % self._stack.shape[0]]
            img = scale_image(crop_image(img, self.crop), self.scale)
            if img.ndim == 3 and self.color == 'Gray':
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), self.frame_num
            else:
                return img, self.frame_num
        else:
            raise StopIteration

        # if self.restep:
        #     # redo the last step by undoing  the frame_num
        #     if self.direction_fwd and self.frame_num > 0:
        #         self.frame_num -= 1
        #     elif not self.direction_fwd and self.frame_num < self._stack.shape[0]-2:
        #         self.frame_num += 1
        #     self.restep = False


def putText(img, text, row=30, col=10, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1):
    '''
    Put text on the image
    '''
    import math
    FONT_SCALE = 6e-4  # Adjust for larger font size in all images
    THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images

    if type(img) is np.ndarray:
        height, width = img.shape[:2]
        fontScale = min(width, height) * FONT_SCALE
        thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
        row = height - row  #  margin from the bottom
        # col = height//20  #  margin from the left
        cv2.putText(img, text, (col, row), fontFace, fontScale, color, thickness, cv2.LINE_AA)



def cv2_img_show(name, img, win_width=None, flags=None, mode='RGB', normalise=False):
    """ show image in a cv2 namedwindow, you can set the width or height"""
    img = img.astype('uint8')
    # try:
    if not hasattr(cv2_img_show, 'names'):
        cv2_img_show.names = []
    if not name in cv2_img_show.names:
        cv2.namedWindow(name, flags)
        cv2_img_show.names.append(name)
        cv2.namedWindow(name, flags)
        # setting fullscreen make this show on front
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, flags)
        # set window size
        # cv2.setWindowProperty(name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE) 
        if win_width is not None:
            height = int(img.shape[0] * win_width / img.shape[1])
            cv2.resizeWindow(name, win_width, height)



    if mode=='RGB' and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = resize(img, width=width, height=height)
    if normalise:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) # normalise to 0-255
    cv2.imshow(name, img)

def overlay_mask( image, mask, color=(255,255,0), alpha=0.4):
    """
    Overlay the mask on the image.

    Args:
        image (numpy.ndarray): The image.
        mask (numpy.ndarray): The mask.

    Returns:
        numpy.ndarray: The image with the mask overlayed.
    """
    # convert to rgb color
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_rgb = np.where(mask_rgb == [255, 255, 255], np.uint8(color), mask_rgb)
    if image.shape[2] == 4:
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2RGBA)

    # resize mask to image size
    mask_rgb = cv2.resize(mask_rgb, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(mask_rgb, alpha, image, 1 , 0)


def draw_bboxes( image, bboxes, confidences, text=True, thickness=6, alpha:typ.Union[float, None]=0.3):
    """
    Draw the bounding boxes about detected objects in the image. Assums the bb are sorted by confidence

    Args:
        image (numpy.ndarray): Image or video frame.
        bboxes (numpy.ndarray): Bounding boxes pixel coordinates as (xmin, ymin, width, height)

    Returns:
        numpy.ndarray: image with the bounding boxes drawn on it.
    """

    # support for alpha blending overlay
    overlay = image.copy() if alpha is not None else image    
    count = 0
    height, width = image.shape[:2]
    for bb in bboxes:
        l, t, w, h = bb
        clr = (255, 0, 0) if count < 5 else (0,255,0) if count < 10 else (0, 0, 255) # in order red, green, blue
        # convert to int and mult by width and height
        l, t, w, h = int(l*width), int(t*height), int(w*width), int(h*height)

        cv2.rectangle(overlay, (l, t ), (l+w, t+h), clr, thickness)

        if text:
            putlabel(overlay, f'{count}', (l,t), fontScale=1.0, color=clr, thickness=2)
        count += 1
    
    if alpha is not None:
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def putlabel(img, text, position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2):    
    """
    Put text on the image with a black background.

    Args:
        img (numpy.ndarray): The image.
        text (str): The text to put on the image.
        position (tuple): The position where the text should be put.
        fontFace (int): The font type. Default is cv2.FONT_HERSHEY_SIMPLEX.
        fontScale (float): Font scale. Default is 1.
        color (tuple): Font color. Default is white.
        thickness (int): Thickness of the lines used to draw a text. Default is 2.

    Returns:
        numpy.ndarray: The image with the text.
    """
    # Calculate the width and height of the text
    (text_width, text_height), baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
    # Determine the y-coordinate for the text
    y_text = max(position[1], text_height)
    # Draw a filled rectangle on the image at the location where the text will be placed
    cv2.rectangle(img, (position[0], y_text - text_height), (position[0] + text_width, y_text + baseLine), (0, 0, 0), cv2.FILLED)
    # Draw the text on the image at the specified location
    cv2.putText(img, text, (position[0], y_text+text_height//4), fontFace, fontScale, color, thickness, cv2.LINE_AA)

    return img


class VideoWriter:
    def __init__(self, filename='_autoplay.mp4', fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.params['filename'] == '_autoplay.mp4':
            self.show()

    def show_video(self, **kw):
        """ Display the video in a jupyter notebook"""
        try:
            import moviepy.editor as mvp
            self.close()
            fn = self.params['filename']
            display(mvp.ipython_display(fn, **kw))   # display in jupyter notebook 
        except Exception as e:
            print(f'Error: {e} not found')

if __name__ == '__main__':
    from skimage.measure import block_reduce

    (rows, cols) = (2000, 3000)
    center = (2750, 4350)
    (_r, _c) = (center[0]-rows//2, center[1]-cols//2)
    crop = [_r, _r + rows, _c, _c + cols]
    home = str(Path.home())
    images = ImageLoader(home+"/data/maui-data/large_plane/images.npy", crop=crop, scale=0.5, color='RGB')
    wait_timeout = 100
    for img, i in images:
        # cmo =  update(cmo)
        # img = next(images)
        # img = resize(img, width=500)
        putText(img, f'Frame = {i}, fontScale=0.5')
        cv2.imshow('image',  img)

        start_time = default_timer()
        minpool = min_pool(img, 12, 12)
        print(f'minpool elapsed time {default_timer() - start_time}')
        start_time = default_timer()
        minpool = block_reduce(img, block_size=(12, 12), func=np.median)
        print(f'block_reduce elapsed time {default_timer() - start_time}')

        cv2.imshow('minpool',  minpool)
        k = cv2.waitKey(wait_timeout)
        if k == ord('q') or k == 27:
            break
        if k == ord(' '):
            wait_timeout = 0
        if k == ord('d'):
            wait_timeout = 0
            images.direction_fwd = not images.direction_fwd
        if k == ord('g'):
            wait_timeout = 100
        if k == ord('r'):
            # change direction
            wait_timeout = 0
            images.restep = True

    cv2.waitKey(1000)
    cv2.destroyAllWindows()

