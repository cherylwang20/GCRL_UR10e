import cv2
import numpy as np

# Read the image
image = cv2.imread('image_rbf.png')

# Convert the image to HSV color space
blurred = cv2.GaussianBlur(image, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# Get the range of colors from the user
lower_bound = np.array([80, 50, 20], dtype=np.uint8)
upper_bound = np.array([100, 255, 255], dtype=np.uint8)

# Create the color mask
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Apply the color mask to the image
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# Show the original and segmented images
cv2.imshow("Original Image", hsv)
cv2.imshow("Segmented Image", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()