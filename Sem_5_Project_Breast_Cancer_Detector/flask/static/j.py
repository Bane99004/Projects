import cv2
import numpy as np

# Load the image
image = cv2.imread('.png')

# Create a blank mask with the same dimensions as the image (all zeros)
mask = np.zeros(image.shape[:2], dtype="uint8")

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range for masking (e.g., masking a blue object)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Create a mask for the blue color
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Apply the mask to the image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# display the image using cv2.imshow()
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', masked_image)