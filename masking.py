import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('image_2.png')
cv2.imshow('raw', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply histogram equalization to enhance contrast
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Apply adaptive thresholding to minimize shadow effects
contrast_img = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
# Use Canny edge detection on the thresholded image
cv2.imshow('contrast', contrast_img)
edges = cv2.Canny(contrast_img, 50, 150)

# Dilate the edges to close gaps
dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

# Find contours from the dilated image
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the beaker is the largest object in the image
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the beaker
mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

# Optional: Apply a mask to the original image to highlight the beaker
result = cv2.bitwise_and(image, image, mask=mask)

# Save or display the results
cv2.imwrite('beaker_mask.png', mask)
cv2.imwrite('beaker_result.png', result)
cv2.imshow('Beaker Mask', mask)
#cv2.imshow('Segmented Beaker', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
