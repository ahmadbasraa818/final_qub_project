#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing necessary modules
import os
import argparse
import logging
import cv2
import numpy

# Setting up logging
logger = logging.getLogger('main')


def get_logger(level=logging.INFO, quite=False, debug=False, to_file=''):
    """
    Initialize a logger to stdout.

    Parameters:
    - level: Logging level (DEBUG, INFO, WARNING, CRITICAL)
    - quite: Boolean to suppress logger output
    - debug: Boolean to set logger to debug level
    - to_file: Filepath to save log output (optional)

    Returns:
    - logger: Initialized logger
    """
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


def get_args(default=None, args_string=''):
    """
    Parse command line arguments.

    Parameters:
    - default: Dictionary of default arguments with keys as `dest`
    - args_string: String of command line arguments (optional)

    Returns:
    - args: Parsed command line arguments
    """
    if not default:
        default = {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_paths', type=str, nargs='+', help="Filepath for input images or folder containing images")
    parser.add_argument('-p', '--superpixel', dest='superpixel', action='store_true',
                        help='generate blur estimation for superpixels')
    parser.add_argument('-r', '--thresh', dest='thresh', default=10, type=str,
                        help='threshold for deciding if blurry (between 0 & 1)')
    parser.add_argument('-m', '--mask', dest='mask', action='store_true',
                        help='Conduct SLIC Segmentation to generate focus mask')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help='display image steps')
    parser.add_argument('-e', '--debug', dest='debug', action='store_true', help='set logger to debug')
    parser.add_argument('-q', '--quite', dest='quite', action='store_true', help='silence the logger')
    parser.add_argument('-s', '--save', dest='save', action='store_true', help='save blur masks')
    parser.add_argument('-t', '--testing', dest='testing', action='store_true', help='enable testing method')
    if args_string:
        args_string = args_string.split(' ')
        args = parser.parse_args(args_string)
    else:
        args = parser.parse_args()
    return args


def gen_args():
    """
    Generate command line arguments.

    Returns:
    - args: Generated command line arguments
    """
    return get_args(args_string='USED_GEN_ARGS')


def find_images(path, recursive=True):
    """
    Find images in a directory.

    Parameters:
    - path: Directory path or file path
    - recursive: Boolean indicating whether to search recursively (default: True)

    Returns:
    - List of image file paths
    """
    if os.path.isdir(path):
        return list(xfind_images(path, recursive=recursive))
    elif os.path.exists(path):
        return [path]
    else:
        raise ValueError('path is not a valid path or directory')


def xfind_images(directory, recursive=False, ignore=True):
    """
    Generator function to find images in a directory.

    Parameters:
    - directory: Directory path
    - recursive: Boolean indicating whether to search recursively (default: False)
    - ignore: Boolean indicating whether to ignore files with '-' in their names (default: True)

    Yields:
    - Image file paths
    """
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
    """
    Display an image with an adjustable scale.

    Parameters:
    - title: Title for the displayed image window
    - img: Numpy array representing the image
    - max_size: Maximum size for the displayed image (default: 200000)
    """
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    scale = numpy.sqrt(min(1.0, float(max_size)/(img.shape[0]*img.shape[1])))
    logger.debug('image is being scaled by a factor of {0}'.format(scale))
    shape = (int(scale*img.shape[1]), int(scale*img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)
