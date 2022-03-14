import cv2
import numpy as np
from contoursBasedMatch import getContours
from contoursBasedMatch import contourMatch
from imageLoadClean import load2whiteblack
import time

def display_np(img_np, title="default"):
    cv2.imshow(title,img_np)
    c = cv2.waitKey()
def display_contour(panel_widht,panel_height,contour_list, contour_idx, title="default"):
    contour_np = np.zeros((panel_widht,panel_height, 3), dtype=np.uint8)
    cv2.drawContours(contour_np, contour_list, contour_idx, (0,255,0), 3)
    cv2.imshow(title,contour_np)
    c = cv2.waitKey()

def debug_method(x,y,target_shape_white_black,target_shape_contour,input_white_black,input_shape_contours,best_matched_contour_idx ):
    display_np(target_shape_white_black, 'target object')
    display_np(input_white_black, 'select panel')
    display_contour(
        target_shape_white_black.shape[0],
        target_shape_white_black.shape[1],
        [target_shape_contour], 
        0, 
        title="Target object contour"
    )
    for i in range(len(input_shape_contours)):
        display_contour(
            input_white_black.shape[0],
            input_white_black.shape[1],
            input_shape_contours,
            i,
            title="Select panel contour " + str(i)
        )
    print("target_shape_contour shape has size: ", len(target_shape_contour), ' shape ', target_shape_contour.shape)
    print("best_matched_contour_idx shape has size: ", len(input_shape_contours[best_matched_contour_idx]))
    imput_image_new = np.zeros((input_white_black.shape[0],input_white_black.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(imput_image_new, input_shape_contours, best_matched_contour_idx, (0,255,0), 3)
    cv2.imshow('Best match object at idx '+str(best_matched_contour_idx),imput_image_new)
    c = cv2.waitKey()
    print("Central point pixel is: ", (x,y))

    imput_image_circle = np.zeros((input_white_black.shape[0],input_white_black.shape[1], 3), dtype=np.uint8)
    # Radius of circle
    radius = 10
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    circle_image = cv2.circle(imput_image_circle,(x,y),radius, color, thickness)

    # Displaying the image
    cv2.imshow("Click inside red circle", circle_image)
    c = cv2.waitKey()
    # match shapes: https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html#:~:text=OpenCV%20comes%20with%20a%20function,on%20the%20hu%2Dmoment%20values.
    ## --------- DONE ------------ ##

# API entry point
# path can be local file path or url path
def geetest(target_object_image_path, select_panel_image_path, debug=False):
    target_shape_white_black = load2whiteblack(target_object_image_path)
    input_white_black = load2whiteblack(select_panel_image_path)
    target_shape_contours=getContours(target_shape_white_black)
    input_shape_contours=getContours(input_white_black)
 
    list_contours = list(target_shape_contours)
    single_contour = []
    for c in list_contours:
        single_contour.extend(c) 
    single_contour = np.array(single_contour)
    print('target single contour shape: ', single_contour.shape)
    best_matched_contour_idx, best_matched_contour = contourMatch(single_contour,input_shape_contours)
    central_point_pixel = np.mean(best_matched_contour, axis=0)
    x = int(central_point_pixel[0][0])
    y = int(central_point_pixel[0][1])
    if debug:
        debug_method(
            x=x,
            y=y,
            target_shape_white_black=target_shape_white_black,
            target_shape_contour=single_contour,
            input_white_black=input_white_black,
            input_shape_contours=input_shape_contours,
            best_matched_contour_idx=best_matched_contour_idx)
    return (x,y)

dir="testcase2"
img1_path="./resources/"+dir + "/geetestplan.jpeg"
img2_path="./resources/"+dir + "/geetestObj2.png"
t_start = time.time()
(x,y) = geetest(img2_path, img1_path, debug=False )
t_end = time.time()
print('Pls click at x: ', x, ' y: ',y)
print('Time cost in second : ', t_end - t_start)

#img_np = readFromUrl('https://static.geetest.com/nerualpic/v4_pic/click_2021_06_16/icon/b8c789ae5a884e69a2926835aa895ede.jpg')



