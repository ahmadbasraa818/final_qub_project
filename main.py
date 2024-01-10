#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, argparse, cv2, os, numpy, FocusMask, matplotlib.pyplot as plt, easygui, glob
import subprocess

logger = logging.getLogger('main')
def run_cpp_executable(img_path):
    # Path to the compiled C++ executable
    cpp_executable = "./your_cpp_executable"  # Replace with the actual path to your compiled C++ executable

    # Run the C++ executable with the image path as an argument
    subprocess.run([cpp_executable, img_path])
class DragAndDropGUI:
    def __init__(self):
        self.image_paths = []

    def drag_and_drop(self):
        message = "Drag and drop images to test blur"
        self.image_paths.extend(easygui.fileopenbox(msg=message, title="Image Blur Tester", default="*.png;*.jpg", multiple=True) or [])

def get_logger(level=logging.INFO, quiet=False, debug=False, to_file=''):
    assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.CRITICAL]
    logger = logging.getLogger('main')
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    if debug:
        level = logging.DEBUG
    logger.setLevel(level=level)
    if not quiet:
        handler = logging.FileHandler(to_file) if to_file else logging.StreamHandler()
        handler.setLevel(level=level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def find_images(path, recursive=True):
    if os.path.isdir(path):
        return list(xfind_images(path, recursive=recursive))
    elif '*' in path or '?' in path:
        return glob.glob(path)
    elif os.path.exists(path):
        return [path]
    else:
        raise ValueError('path is not a valid path or directory')

def xfind_images(directory, recursive=False, ignore=True):
    assert os.path.isdir(directory), 'FileIO - get_images: Directory does not exist'
    assert isinstance(recursive, bool), 'FileIO - get_images: recursive must be a boolean variable'
    ext, result = ['png', 'jpg', 'jpeg'], []
    for path_a in os.listdir(directory):
        path_a = directory+'/'+path_a
        if os.path.isdir(path_a) and recursive:
            for path_b in xfind_images(path_a):
                yield path_b
        check_a = path_a.split('.')[-1] in ext
        check_b = ignore or ('-' not in path_a.split('/')[-1])
        if check_a and check_b:
            yield path_a

def display(title, img, max_size=200000):
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    scale = numpy.sqrt(min(1.0, float(max_size)/(img.shape[0]*img.shape[1])))
    logger.debug('image is being scaled by a factor of {0}'.format(scale))
    shape = (int(scale*img.shape[1]), int(scale*img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)

def evaluate(img_col, args):
    numpy.seterr(all='ignore')
    assert isinstance(img_col, numpy.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)
    assert isinstance(args, argparse.Namespace), 'args must be of type argparse.Namespace not {0}'.format(type(args))

    img_gry = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gry.shape
    crow, ccol = rows // 2, cols // 2

    f = numpy.fft.fft2(img_gry)
    fshift = numpy.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = numpy.fft.ifftshift(fshift)
    img_fft = numpy.fft.ifft2(f_ishift)
    img_fft = 20 * numpy.log(numpy.abs(img_fft))

    if args.display and not args.testing:
        cv2.destroyAllWindows()
        display('img_fft', img_fft)
        display('img_col', img_col)
        cv2.waitKey(0)

    result = numpy.mean(img_fft)
    return img_fft, result, result < args.thresh

def blur_detector(img_col, thresh=10, mask=False):
    assert isinstance(img_col, numpy.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)

    args = argparse.Namespace(image_paths=[], superpixel=False, thresh=thresh, mask=False, display=False, debug=False,
                              quiet=False, save=False, testing=False)

    if mask:
        return FocusMask.blur_mask(img)
    else:
        return evaluate(img_col=img_col, args=args)

if __name__ == '__main__':
    args = {'image_paths': [], 'superpixel': True, 'thresh': 10, 'mask': True, 'display': True, 'debug': False, 'quiet': False, 'save': False, 'testing': False}
    drag_and_drop_gui = DragAndDropGUI()
    drag_and_drop_gui.drag_and_drop()
    args['image_paths'] = drag_and_drop_gui.image_paths
    logger = get_logger(quiet=False, debug=False)

    image_paths = args['image_paths']

    for path in image_paths:
        logger.debug(f'Evaluating {path}')
        img = cv2.imread(path)

        # Continue with your existing image processing logic...

    x_okay, y_okay = [], []
    x_blur, y_blur = [], []

    for path in args['image_paths']:
        for img_path in find_images(path):
            logger.debug('evaluating {0}'.format(img_path))
            img = cv2.imread(img_path)

            if isinstance(img, numpy.ndarray):
                if args['testing']:
                    display('dialog (blurry: Y?)', img)
                    blurry = False
                    if cv2.waitKey(0) in map(lambda i: ord(i), ['Y', 'y']):
                        blurry = True

                    x_axis = [1, 3, 5, 7, 9]
                    for x in x_axis:
                        img_mod = cv2.GaussianBlur(img, (x, x), 0)
                        y = evaluate(img_mod, args={'display': args['display'], 'testing': args['testing']})[0]
                        if blurry:
                            x_blur.append(x)
                            y_blur.append(y)
                        else:
                            x_okay.append(x)
                            y_okay.append(y)
                elif args['mask']:
                    msk, res, blurry = FocusMask.blur_mask(img)
                    img_msk = cv2.bitwise_and(img, img, mask=msk)
                    if args['display']:
                        display('res', img_msk)
                        display('msk', msk)
                        display('img', img)
                        cv2.waitKey(0)
                else:
                    img_fft, result, val = evaluate(img, args=args)
                    logger.info('fft average of {0}'.format(result))

                    if args['display']:
                        display('input', img)
                        display('img_fft', img_fft)
                        cv2.waitKey(0)

    if args['display'] and args['testing']:
        logger.debug('x_okay: {0}'.format(x_okay))
        logger.debug('y_okay: {0}'.format(y_okay))
        logger.debug('x_blur: {0}'.format(x_blur))
        logger.debug('y_blur: {0}'.format(y_blur))
        plt.scatter(x_okay, y_okay, color='g')
        plt.scatter(x_blur, y_blur, color='r')
        plt.grid(True)
        plt.show()
