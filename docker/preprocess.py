import os, sys
import glob
import cv2
import numpy as np
import copy
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
 
def calculate_WBC_radius(image, prct_reduced = 1): 
    # convert to gray scale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply median filter for smoothning
    blurM = cv2.medianBlur(gray, 5)

    # apply gaussian filter for smoothning
    blurG = cv2.GaussianBlur(gray, (9, 9), 0)

    # histogram equalization
    histoNorm = cv2.equalizeHist(gray)

    # create a CLAHE object for
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8))
    claheNorm = clahe.apply(gray)



    # contrast stretching
    # Function to map each intensity level to output intensity level.
    def pixelVal(pix, r1, s1, r2, s2):
        if (0 <= pix and pix <= r1):
            return (s1 / r1) * pix
        elif (r1 < pix and pix <= r2):
            return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

        # Define parameters.


    r1 = 70
    s1 = 0
    r2 = 200
    s2 = 255

    # Vectorize the function to apply it to each value in the Numpy array.
    pixelVal_vec = np.vectorize(pixelVal)

    # Apply contrast stretching.
    contrast_stretched = pixelVal_vec(gray, r1, s1, r2, s2)
    contrast_stretched_blurM = pixelVal_vec(blurM, r1, s1, r2, s2)


    # edge detection using canny edge detector
    edge = cv2.Canny(gray, 100, 200)

    edgeG = cv2.Canny(blurG, 100, 200)

    edgeM = cv2.Canny(blurM, 100, 200)

    # read enhanced image
    img = edgeM

    # morphological operations
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 1)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Adaptive thresholding on mean and gaussian filter
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                cv2.THRESH_BINARY, 11, 2)
    # Otsu's thresholding
    ret4, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # hough transform with modified circular parameters
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 20,
                               param1 = int(148* prct_reduced), 
                               param2 = int(83* prct_reduced), 
                               minRadius = 1, 
                               maxRadius = int(148 * prct_reduced))
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    
    Cell_count = [circle[-1] for circle in circles]
    print(circles)
    return int(np.mean(Cell_count))

def circle_crop(img):
    '''returns a cropped image according to the circle as well as the removed top part and removed left part of the image'''

    # select circle only
    img_gray = rgb2gray(img)
    th = threshold_otsu(img_gray)
    fg = img_gray>th
    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(fg))
    min_coords = np.min(coords, axis=1) # [y1, x1]
    max_coords = np.max(coords, axis=1) # [y2, x2]
    height, width, c = img.shape
    
    ''' image[start_row:end_row, start_column:end_column] e.g. image[30:250, 100:230] or [x1:x2, y1:y2]
    You can see that the waterfall goes vertically starting at about 30px and ending at around 250px.
    You can see that the waterfall goes horizontally from around 100px to around 230px. 
                '''

    img_cropped = img[min_coords[0]:max_coords[0],
                min_coords[1]:max_coords[1]]

    xmin = min_coords[0] / width
    xmax = max_coords[0] / width
    ymin = min_coords[1] / height
    ymax = max_coords[1] / height

    return img_cropped, xmin, xmax, ymin, ymax