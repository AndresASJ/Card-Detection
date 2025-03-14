import cv2
import os

def load_images_from_directory(directory, extension=('.png', '.jpg')):
    """
    Loads the template images we're going to be comparing every frame to
    """
    images = []  # List to store loaded images
    for file_name in os.listdir(directory):  # Iterate through all files in the directory
        if file_name.endswith(extension):  # Check if the file has the specified extension
            image_path = os.path.join(directory, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:  # Check if the image was successfully loaded
                image = cv2.resize(image, (300, 400))  # resizing to a smaller image helps with faster processing & consistency
                images.append((file_name, image))  # Add the image and its filename to the list
            else:
                print(f"Error: Could not load {file_name}.")
    return images

def preprocess_frame(frame):
    """
    Preprocesses a frame for card detection
    """
    # Convert to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    
    # Apply adaptive thresholding with inverted binary
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    return binary
