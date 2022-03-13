import cv2
import numpy as np

# https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
def getContours(img_white_black, idx=-1):
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
    # RETR_CCOMP: retrieves all of the contours and organizes them into a two-level hierarchy.
    #  At the top level, there are external boundaries of the components. 
    # At the second level, there are boundaries of the holes. 
    # If there is another contour inside a hole of a connected component, it is still put at the top level.

    # hierarchy + cv2.RETR_TREE can tell you parent-child relationship between contours:
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
    contours, hierarchy= cv2.findContours(img_white_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_imag = np.empty((img_white_black.shape[0],img_white_black.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(result_imag, contours, -1, (0,255,0), 3)
    return result_imag,contours

# retrun best matched contours and index of it in contours_list
def contourMatch(target_countours, contours_list):
    score,ct,idx=1,None,0
    for k in range(len(contours_list)):
        ret = cv2.matchShapes(target_countours,contours_list[k],1,0.0)
        if ret <score:
            score = ret
            ct = ret
            idx=k
    return idx, ct
