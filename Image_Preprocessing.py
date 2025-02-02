import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

def preprocess_image(image_path, output_size=(128, 128), enhance_contrast=True, flip=True, normalize=True):
    """
    Perform comprehensive preprocessing on input images, including cropping, resizing, flipping, 
    contrast enhancement, and normalization.

    Parameters:
        image_path (str): Path to the input image
        output_size (tuple): Desired output image size (width, height)
        enhance_contrast (bool): Whether to enhance image contrast
        flip (bool): Whether to apply random flipping
        normalize (bool): Whether to normalize pixel values

    Returns:
        processed_image (numpy.ndarray): The preprocessed image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Cropping: Center crop the image to a square
    h, w, _ = image.shape
    crop_size = min(h, w)
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # Resizing: Resize the cropped image to the desired size
    resized_image = cv2.resize(cropped_image, output_size)


    processed_image = Image.fromarray(resized_image)
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(1.5)  # Contrast enhancement factor
    if flip and random.random() > 0.5:
        processed_image = processed_image.transpose(Image.FLIP_LEFT_RIGHT)
    processed_image = np.array(processed_image)
    if normalize:
        processed_image = processed_image / 255.0

    return processed_image


if __name__ == "__main__":
    
    image_paths = ["raw_image_1.jpg", "raw_image_2.jpg", "raw_image_3.jpg", "raw_image_4.jpg", "raw_image_5.jpg"]

    for i, image_path in enumerate(image_paths):
        output_image = preprocess_image(image_path)

        output_path = f"processed_image_{i+1}.jpg"
        processed_image_uint8 = (output_image * 255).astype(np.uint8) 
        Image.fromarray(processed_image_uint8).save(output_path)

        print(f"Processed image saved to {output_path}")
