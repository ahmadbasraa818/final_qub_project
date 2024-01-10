#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing necessary modules
import logging
import argparse
import cv2
import os
import numpy
import FocusMask  # Custom module for creating a mask
import matplotlib.pyplot as plt

# Setting up logging
logger = logging.getLogger('main')

def get_logger(level=logging.INFO, quite=False, debug=False, to_file=''):
    assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.CRITICAL]
    logger = logging.getLogger('main')
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    if debug:
        level = logging.DEBUG
    logger.setLevel(level=level)
    if not quite:
        if to_file:
            fh = logging.FileHandler(to_file)
            fh.setLevel(level=level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(level=level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    return logger

def find_images(path, recursive=True):
    if os.path.isdir(path):
        return list(xfind_images(path, recursive=recursive))
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

# Function to evaluate blur using FFT
def evaluate(img_col, args):
    numpy.seterr(all='ignore')
    assert isinstance(img_col, numpy.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)
    assert isinstance(args, argparse.Namespace), 'args must be of type argparse.Namespace not {0}'.format(type(args))

    # Convert the color image to grayscale
    img_gry = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gry.shape
    crow, ccol = rows // 2, cols // 2  # Use // for integer division

    # Apply FFT
    f = numpy.fft.fft2(img_gry)
    fshift = numpy.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = numpy.fft.ifftshift(fshift)
    img_fft = numpy.fft.ifft2(f_ishift)
    img_fft = 20 * numpy.log(numpy.abs(img_fft))

    # Display the results if specified
    if args.display and not args.testing:
        cv2.destroyAllWindows()
        display('img_fft', img_fft)
        display('img_col', img_col)
        cv2.waitKey(0)

    # Calculate the mean of the FFT result
    result = numpy.mean(img_fft)

    # Return the FFT result, mean value, and whether it's below the specified threshold
    return img_fft, result, result < args.thresh

# Function to detect blur in an image
def blur_detector(img_col, thresh=10, mask=False):
    assert isinstance(img_col, numpy.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)

    # Generate argument namespace
    args = argparse.Namespace(image_paths=[], superpixel=False, thresh=thresh, mask=False, display=False, debug=False,
                              quite=False, save=False, testing=False)

    # Either create a blur mask or evaluate blur using FFT
    if mask:
        return FocusMask.blur_mask(img)
    else:
        return evaluate(img_col=img_col, args=args)

# Main script execution
if __name__ == '__main__':
    args = {
        'image_paths': ["ImagesToTest"],  # Specify the path to your images
        'superpixel': True,
        'thresh': 10,
        'mask': True,
        'display': True,
        'debug': False,
        'quite': False,
        'save': False,
        'testing': False
    }

    # Configure logging
    logger = get_logger(quite=args['quite'], debug=args['debug'])

    # Lists to store data for plotting in testing mode
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