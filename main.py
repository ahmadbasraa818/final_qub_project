#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing necessary modules
import logging
import argparse
import cv2
import numpy
import scripts  # Custom utility functions
import FocusMask  # Custom module for creating a mask

# Setting up logging
logger = logging.getLogger('main')

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
        scripts.display('img_fft', img_fft)
        scripts.display('img_col', img_col)
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
    args = scripts.gen_args()
    args.thresh = thresh

    # Either create a blur mask or evaluate blur using FFT
    if mask:
        return FocusMask.blur_mask(img)
    else:
        return evaluate(img_col=img_col, args=args)

# Main script execution
if __name__ == '__main__':
    # Parse command-line arguments
    args = scripts.get_args()

    # Configure logging
    logger = scripts.get_logger(quite=args.quite, debug=args.debug)

    # Lists to store data for plotting in testing mode
    x_okay, y_okay = [], []
    x_blur, y_blur = [], []

    # Iterate through specified image paths
    for path in args.image_paths:
        for img_path in scripts.find_images(path):
            logger.debug('evaluating {0}'.format(img_path))
            img = cv2.imread(img_path)

            # Check if the read image is a valid numpy array
            if isinstance(img, numpy.ndarray):
                if args.testing:
                    # Allow the user to manually label images as blurry or not
                    scripts.display('dialog (blurry: Y?)', img)
                    blurry = False
                    if cv2.waitKey(0) in map(lambda i: ord(i), ['Y', 'y']):
                        blurry = True

                    # For testing, apply blur and collect data for plotting
                    x_axis = [1, 3, 5, 7, 9]
                    for x in x_axis:
                        img_mod = cv2.GaussianBlur(img, (x, x), 0)
                        y = evaluate(img_mod, args=args)[0]
                        if blurry:
                            x_blur.append(x)
                            y_blur.append(y)
                        else:
                            x_okay.append(x)
                            y_okay.append(y)
                elif args.mask:
                    # If mask is specified, create a blur mask and display images
                    msk, res, blurry = FocusMask.blur_mask(img)
                    img_msk = cv2.bitwise_and(img, img, mask=msk)
                    if args.display:
                        scripts.display('res', img_msk)
                        scripts.display('msk', msk)
                        scripts.display('img', img)
                        cv2.waitKey(0)
                else:
                    # Evaluate blur using FFT and log the result
                    img_fft, result, val = evaluate(img, args=args)
                    logger.info('fft average of {0}'.format(result))

                    # Display the original and FFT images if specified
                    if args.display:
                        scripts.display('input', img)
                        scripts.display('img_fft', img_fft)
                        cv2.waitKey(0)

    # Display the scatter plot of data in testing mode
    if args.display and args.testing:
        import matplotlib.pyplot as plt
        logger.debug('x_okay: {0}'.format(x_okay))
        logger.debug('y_okay: {0}'.format(y_okay))
        logger.debug('x_blur: {0}'.format(x_blur))
        logger.debug('y_blur: {0}'.format(y_blur))
        plt.scatter(x_okay, y_okay, color='g')
        plt.scatter(x_blur, y_blur, color='r')
        plt.grid(True)
        plt.show()
