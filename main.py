import cv2
import numpy as np
from contoursBasedMatch import getContours
from contoursBasedMatch import contourMatch
from imageLoadClean import load2whiteblack

def geetest(target_object_image_path, select_panel_image_path):
    target_shape_white_black = load2whiteblack(target_object_image_path)
    input_white_black = load2whiteblack(select_panel_image_path)
    result_target_shape_img,target_shape_contours=getContours(target_shape_white_black)
    result_imag2,input_shape_contours=getContours(input_white_black)
    idx, ctx = contourMatch(target_shape_contours[0],input_shape_contours)
    central_point_pixel = np.mean(ctx, axis=0)
    x = central_point_pixel[0][0]
    y = central_point_pixel[0][1]
    return (x,y), target_shape_white_black,target_shape_contours,input_white_black,input_shape_contours,idx,ctx


(x,y),target_shape_white_black,target_shape_contours,input_white_black,input_shape_contours,idx,ctx = geetest("./resources/geetestObj3.png","./resources/geetestplan.jpeg" )


## --------- DEBUG ----------- ##
cv2.imshow('target_shape_white_black',target_shape_white_black)
c = cv2.waitKey()
cv2.imshow('input_white_black',input_white_black)
c = cv2.waitKey()
print("target_shape_contours shape has size: ", len(target_shape_contours))
print("input_shape_contours shape has size: ", len(input_shape_contours))
imput_image_new = np.empty((input_white_black.shape[0],input_white_black.shape[1], 3), dtype=np.uint8)
cv2.drawContours(imput_image_new, input_shape_contours, idx, (0,255,0), 3)
cv2.imshow('imput_image_new',imput_image_new)
c = cv2.waitKey()
# match shapes: https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html#:~:text=OpenCV%20comes%20with%20a%20function,on%20the%20hu%2Dmoment%20values.
## --------- DONE ------------ ##

print("Object pixel position in input image:", len(ctx))
print("Central point pixel is: ", (x,y))


imput_image_circle = np.empty((input_white_black.shape[0],input_white_black.shape[1], 3), dtype=np.uint8)
# Radius of circle
radius = 20
  
# Blue color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2

circle_image = cv2.circle(imput_image_circle,(int(x),int(y)),radius, color, thickness)

# Displaying the image
cv2.imshow("circle_image", circle_image)
c = cv2.waitKey()
