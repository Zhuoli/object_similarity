import cv2
import numpy as np
from contoursBasedMatch import getContours
from contoursBasedMatch import contourMatch
from imageLoadClean import load2whiteblack

target_shape_white_black = load2whiteblack("./resources/geetestObj3.png")
input_white_black = load2whiteblack("./resources/geetestplan.jpeg")
result_target_shape_img,target_shape_contours=getContours(target_shape_white_black)
result_imag2,input_shape_contours=getContours(input_white_black)
idx, ctx = contourMatch(target_shape_contours[0],input_shape_contours)
imput_image_new = np.empty((input_white_black.shape[0],input_white_black.shape[1], 3), dtype=np.uint8)


## --------- DEBUG ----------- ##
cv2.imshow('target_shape_white_black',target_shape_white_black)
c = cv2.waitKey()
cv2.imshow('input_white_black',input_white_black)
c = cv2.waitKey()
print("target_shape_contours shape has size: ", len(target_shape_contours))
print("input_shape_contours shape has size: ", len(input_shape_contours))
cv2.drawContours(imput_image_new, input_shape_contours, idx, (0,255,0), 3)
cv2.imshow('imput_image_new',imput_image_new)
c = cv2.waitKey()
# match shapes: https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html#:~:text=OpenCV%20comes%20with%20a%20function,on%20the%20hu%2Dmoment%20values.
## --------- DONE ------------ ##
