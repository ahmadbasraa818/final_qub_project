import logging
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import easygui
import glob
import subprocess
import skimage.segmentation
import skimage.measure

class DragAndDropGUI:
    def __init__(self):
        self.image_paths = []

    def drag_and_drop(self):
        message = "Drag and drop images to test blur"
        selected_paths = easygui.fileopenbox(msg=message, title="Image Blur Tester", default="*.png;*.jpg", multiple=True)
        if selected_paths:
            self.image_paths.extend(selected_paths)

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

def load_fft_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    fft_data = [np.array([complex(float(val.split(',')[0]), float(val.split(',')[1])) for val in line.split()]) for line in lines]
    return np.array(fft_data)

def display(title, img, max_size=200000):
    scale = np.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_blurriness_ratio(fft_data):
    magnitude_spectrum = np.abs(fft_data)
    total_energy = np.sum(magnitude_spectrum)
    # Assuming high frequencies are in the corners of the FFT matrix
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    # Define a high-frequency region (e.g., outer 25% of the spectrum)
    hfreq_mask = np.zeros_like(magnitude_spectrum)
    hfreq_mask[:crow//2, :] = 1
    hfreq_mask[(crow//2)*3:, :] = 1
    hfreq_mask[:, :ccol//2] = 1
    hfreq_mask[:, (ccol//2)*3:] = 1
    high_freq_energy = np.sum(magnitude_spectrum * hfreq_mask)
    return high_freq_energy / total_energy

def create_mask_from_fft(fft_data):
    magnitude = 20 * np.log(np.abs(fft_data))
    cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.array(magnitude, dtype=np.uint8)
    _, mask = cv2.threshold(magnitude, 120, 255, cv2.THRESH_BINARY)
    return mask

def run_fft_analysis(image_paths):
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        # Run the external FFT process
        command = ['./NewFFT', image_path]
        subprocess.run(command, check=True)
        # Load FFT results
        fft_file = image_path + "_fft_results.csv"
        fft_results = load_fft_results(fft_file)
        # Visualize the results
        fft_results = load_fft_results('path_to_fft_results.csv')  # Load your FFT results here
        visualize_blurriness_heatmap(fft_results)
        blurriness_ratio = calculate_blurriness_ratio(fft_results)
        print(f"Blurriness ratio (lower indicates blurrier): {blurriness_ratio:.4f}")
        # Optionally, display a mask or the original image for visual inspection
        img = cv2.imread(image_path)
        if img is not None:
            display("Processed Image", img)



def visualize_blur_via_ifft(fft_data, image_path):
    # Apply a low-pass filter to the FFT data
    rows, cols = fft_data.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros_like(fft_data)
    mask_size = min(rows, cols) // 5  # Example size, can be tuned
    mask[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 1
    low_pass_fft_data = fft_data * mask

    # Perform an inverse FFT
    ifft_result = np.fft.ifft2(np.fft.ifftshift(low_pass_fft_data))
    ifft_image = np.abs(ifft_result)

    # Load the original image for comparison
    original_image = cv2.imread(image_path, 0)
    original_image_scaled = cv2.resize(original_image, (cols, rows))

    # Normalize and display the images
    ifft_image_normalized = cv2.normalize(ifft_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    original_image_normalized = cv2.normalize(original_image_scaled, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(original_image_normalized, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(ifft_image_normalized, cmap='gray')
    plt.title('Blurriness Visualization')
    plt.axis('off')

    plt.show()

def visualize_blurriness_heatmap(fft_data, blurriness_ratio):
    rows, cols = fft_data.shape
    crow, ccol = rows // 2, cols // 2
    
    # Here's an example usage of blurriness_ratio to adjust the mask size dynamically
    mask_size = int((crow + ccol) // 4 * blurriness_ratio)  # This is just an example
    
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask = np.exp(-(x*x + y*y) / (2.0 * mask_size**2))
    
    fft_magnitude = np.abs(fft_data)
    low_freq_magnitude = fft_magnitude * mask
    log_low_freq_magnitude = np.log1p(low_freq_magnitude)
    
    norm_low_freq_mag = cv2.normalize(log_low_freq_magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(norm_low_freq_mag, cmap='inferno')
    plt.title('Blurriness Heatmap')
    plt.axis('off')
    plt.colorbar()
    plt.show()


def main():
    selected_paths = easygui.fileopenbox(msg="Drag and drop images to test blur", 
                                         title="Image Blur Tester", 
                                         default="*.png;*.jpg", 
                                         multiple=True)
    if not selected_paths:
        print("No images selected.")
        return
    
    # Resolve wildcards to actual file paths and remove any duplicates
    resolved_paths = set()
    for path in selected_paths:
        if '*' in path or '?' in path:
            resolved_paths.update(glob.glob(path))
        else:
            resolved_paths.add(path)
    
    for img_path in resolved_paths:
        print(f"Processing image: {img_path}")
        # Run the external FFT process
        subprocess.run(['./NewFFT', img_path], check=True)
        
        # Load FFT results
        fft_file = img_path + "_fft_results.csv"  # Adjusted to the naming convention
        fft_results = load_fft_results(fft_file)
        
        # Calculate the blurriness ratio
        blurriness_ratio = calculate_blurriness_ratio(fft_results)
        print(f"Blurriness ratio (lower indicates blurrier): {blurriness_ratio:.4f}")
        
        # Visualize the blurriness heatmap
        visualize_blurriness_heatmap(fft_results, blurriness_ratio)
        
        # Optionally, display the original image for visual inspection
        img = cv2.imread(img_path)
        if img is not None:
            display("Processed Image", img)
        else:
            print(f"Failed to load the processed image: {img_path}")

if __name__ == "__main__":
    main()
