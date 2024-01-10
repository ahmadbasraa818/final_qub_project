#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing necessary modules
import logging
import cv2
import numpy
import skimage
import skimage.measure
import skimage.segmentation
import main  # Importing the main module (containing blur_detector function)


# Setting up logging
logger = logging.getLogger('main')


# Function to obtain superpixel masks using SLIC segmentation
def get_masks(img, n_seg=250):
    logger.debug('SLIC segmentation initialized')
    # Perform SLIC segmentation on the image
    segments = skimage.segmentation.slic(img, n_segments=n_seg, compactness=10, sigma=1)
    logger.debug('SLIC segmentation complete')
    logger.debug('contour extraction...')
    masks = [[numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint8), None]]

    # Iterate through superpixels and extract convex hull masks
    for region in skimage.measure.regionprops(segments):
        masks.append([masks[0][0].copy(), region.bbox])
        x_min, y_min, x_max, y_max = region.bbox
        masks[-1][0][x_min:x_max, y_min:y_max] = skimage.img_as_ubyte(region.convex_image)

    logger.debug('contours extracted')
    return masks[1:]


# Function to create a blur mask for the input image
def blur_mask_old(img):
    assert isinstance(img, numpy.ndarray), 'img_col must be a numpy array'
    assert img.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img.ndim)

    # Initialize a blank blur mask
    blur_mask = numpy.zeros(img.shape[:2], dtype=numpy.uint8)

    # Iterate through superpixel masks
    for mask, loc in get_masks(img):
        logger.debug('Checking Mask: {0}'.format(numpy.unique(mask)))
        logger.debug('SuperPixel Mask Percentage: {0}%'.format(int((100.0/255.0)*(numpy.sum(mask)/mask.size))))

        # Evaluate blur for the superpixel region
        img_fft, val, blurry = main.blur_detector(img[loc[0]:loc[2], loc[1]:loc[3]])
        logger.debug('Blurry: {0}'.format(blurry))

        # If blurry, add the superpixel mask to the overall blur mask
        if blurry:
            blur_mask = cv2.add(blur_mask, mask)

    # Calculate the percentage of the image that is blurry
    result = numpy.sum(blur_mask)/(255.0*blur_mask.size)
    logger.info('{0}% of input image is blurry'.format(int(100*result)))
    return blur_mask, result


# Function for morphological operations on a mask
def morphology(msk):
    assert isinstance(msk, numpy.ndarray), 'msk must be a numpy array'
    assert msk.ndim == 2, 'msk must be a greyscale image'

    # Erosion operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    msk = cv2.erode(msk, kernel, iterations=1)

    # Closing operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)

    # Thresholding
    msk[msk < 128] = 0
    msk[msk > 127] = 255

    return msk


# Function to remove a border from a mask
def remove_border(msk, width=50):
    assert isinstance(msk, numpy.ndarray), 'msk must be a numpy array'
    assert msk.ndim == 2, 'msk must be a greyscale image'
    
    # Define border dimensions
    dh, dw = map(lambda i: i//width, msk.shape)
    h, w = msk.shape
    
    # Set the top, bottom, left, and right borders to 255
    msk[:dh, :] = 255
    msk[h-dh:, :] = 255
    msk[:, :dw] = 255
    msk[:, w-dw:] = 255

    return msk


# Function to create a blur mask with additional processing
def blur_mask(img):
    assert isinstance(img, numpy.ndarray), 'img_col must be a numpy array'
    assert img.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img.ndim)

    # Obtain the initial blur mask, its value, and the blur evaluation
    msk, val, blurry = main.blur_detector(img)

    logger.debug('inverting img_fft')
    # Invert the mask and adjust intensity levels
    msk = cv2.convertScaleAbs(255-(255*msk/numpy.max(msk)))
    msk[msk < 50] = 0
    msk[msk > 127] = 255

    logger.debug('removing border')
    # Remove borders from the mask
    msk = remove_border(msk)

    logger.debug('applying erosion and dilation operators')
    # Apply morphological operations
    msk = morphology(msk)

    logger.debug('evaluation complete')
    # Calculate the percentage of the image that is blurry
    result = numpy.sum(msk)/(255.0*msk.size)
    logger.info('{0}% of input image is blurry'.format(int(100*result)))

    return msk, result, blurry


# Main script execution
if __name__ == '__main__':
    # Specify the path to the image
    img_path = "/BlueDectection/demo.png"
    
    # Read the image
    img = cv2.imread(img_path)
    
    # Obtain the blur mask and its value
    msk, val = blur_mask(img)
    
    # Display the original image and the blur mask
    main.display('img', img)
    main.display('msk', msk)
    cv2.waitKey(0)
