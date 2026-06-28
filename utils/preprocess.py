import cv2
import numpy as np

def remove_black_borders(image, threshold = 10):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        return image[y : y + h, x : x + w]
    return image

def preprocess_image(image, target_size = (224, 224)):

    cropped = remove_black_borders(image)
    resized = cv2.resize(cropped, target_size)
    normalized = resized / 255.0

    return normalized, resized