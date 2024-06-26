import cv2
import numpy as np

# Read the image
image = cv2.imread('image_rbf.png')

# Convert the image to HSV color space
#blurred = cv2.GaussianBlur(image, (11, 11), 0)

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contrast_img = cv2.convertScaleAbs(image, alpha=1.3, beta=40)

cv2.imshow('contrast_img', contrast_img)

edges = cv2.Canny(contrast_img, 50, 150)


contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
#cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Edges', edges)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
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
'''