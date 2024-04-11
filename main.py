import logging
import argparse
import cv2
import os
import numpy
import matplotlib.pyplot as plt
import easygui
import glob
import subprocess
import skimage.segmentation
import skimage.measure

logger = logging.getLogger('main')

class DragAndDropGUI:
    def __init__(self):
        self.image_paths = []

    def drag_and_drop(self):
        message = "Drag and drop images to test blur"
        selected_paths = easygui.fileopenbox(msg=message, title="Image Blur Tester", default="*.png;*.jpg", multiple=True)
        if selected_paths:
            self.image_paths.extend(selected_paths[1:])  # Get only selected files, excluding the filter

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

def evaluate(img_col, args, block):
    numpy.seterr(all='ignore')
    assert isinstance(img_col, numpy.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)
    assert isinstance(args, argparse.Namespace), 'args must be of type argparse.Namespace not {0}'.format(type(args))

    img_gry = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gry.shape
    crow, ccol = rows // 2, cols // 2

    f = numpy.fft.fft2(img_gry)
    fshift = numpy.fft.fftshift(f)
    fshift[crow - block:crow + block, ccol - block:ccol + block] = 0
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

def blur_detector(img_col, thresh=10, mask=False, block=100):
    assert isinstance(img_col, numpy.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)

    args = argparse.Namespace(image_paths=[], superpixel=False, thresh=thresh, mask=False, display=False, debug=False,
                              quiet=False, save=False, testing=False)

    if mask:
        return blur_mask(img)
    else:
        return evaluate(img_col=img_col, args=args, block=block)

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
    msk, val, blurry = blur_detector(img)

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

def main():
    args = {'image_paths': [], 'superpixel': True, 'thresh': 10, 'mask': True, 'display': True, 'debug': False, 'quiet': False, 'save': False, 'testing': False}
    drag_and_drop_gui = DragAndDropGUI()
    drag_and_drop_gui.drag_and_drop()
    args['image_paths'] = drag_and_drop_gui.image_paths
    if args['image_paths']:
        folder_path = os.path.dirname(args['image_paths'][0])

        # Prompt user for BLOCK value
        block = int(input("Enter the value of BLOCK (0 - 100): "))
        
        print("press q to quit the windows")
        key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
        if key == ord('q'):  # Check if the pressed key is 'q'
            cv2.destroyAllWindows()  # Close all OpenCV windows

        # Get the precision level from the user
        print(args['image_paths'])
        command = ['./NewFFT'] + args['image_paths']
        subprocess.run(command, text=True)

        # Validate the precision value
             # Call the C++ executable with precision and image paths as inputs
#=========================================================================================================#
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
                    elif args['mask']:
                        msk, res, blurry = blur_mask(img)
                        img_msk = cv2.bitwise_and(img, img, mask=msk)
                        if args['display']:
                            display('res', img_msk)
                            display('msk', msk)
                            display('img', img)
                            cv2.waitKey(0)


#=========================+++++++++==============================================================++#
    image_paths = args['image_paths']


if __name__ == '__main__':
    main()