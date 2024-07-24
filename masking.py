import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200

# Read the image
image = cv.imread('rgb_1.png')
color = "red"
#rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
blurred = cv.GaussianBlur(image, (11, 11), 0)
hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
# we might want to add a series of color to identify e.g., green, blue, red, yellow. 
if color == 'red':
    Lower = (0, 50, 50)
    Upper = (10, 255, 255)
elif color == 'green':
    Lower = (29, 86, 56)
    Upper = (64, 255, 255)
elif color == 'blue':
    Lower = (80, 50, 20)
    Upper = (100, 255, 255)
else:
    raise Warning('please define a valid color (red, gree, blue)')
mask = cv.inRange(hsv, Lower, Upper)
mask = cv.erode(mask, None, iterations=2)
mask = cv.dilate(mask, None, iterations=2)
#print(mask.shape, rgb.shape)

#define the grasping rectangle
x1, y1 = int(53/200 * IMAGE_WIDTH), 0
x2, y2 = int(156/200 * IMAGE_WIDTH), int(68/200 * IMAGE_HEIGHT)

cv.rectangle(image, (x1, 0), (x2, y2), (0, 0, 255), thickness=2)
cv.rectangle(mask, (x1, 0), (x2, y2), 255, thickness=1)

roi = mask[y1:y2, x1:x2]
white_pixels = np.sum(roi == 255)
total_pixels = roi.size
pixel_perc = (white_pixels / total_pixels) * 100
total_pix = (np.sum(mask==255)/mask.size) * 100
print(total_pix)
#print(f"Percentage of white pixels in the rectangle: {self.pixel_perc:.2f}%")

#cv.circle(rgb, (self.cx, self.cy), 5, (255, 0, 0), -1)
#cv.circle(rgb, (105, 34, ), 5, (0, 255, 0), -1)
cv.imshow("rbg", image)# cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
cv.imshow("mask", mask)
#cv.imshow('Inverted Colored Depth', depth_normalized)
cv.waitKey(1)
cv.waitKey(delay=5000)
# cv.destroyAllWindows()
cv.imwrite('mask_1.png', mask)
