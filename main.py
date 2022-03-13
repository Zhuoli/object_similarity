import cv2
import numpy as np
from PIL import Image
import os.path 
from contoursBasedMatch import getContours


def rgba2rgb(rgba_img_numpy):
    print("RGBA_imag_numpy shape: ",rgba_img_numpy.shape)
    a=rgba_img_numpy[:,:,0:3]
    b=rgba_img_numpy[:,:,3]
    c=a + b[:,:,None]
    return c


img1 = cv2.imread(cv2.samples.findFile("./resources/geetestObj3.png"), cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED for PNG file to count transparency layer
img1 = rgba2rgb(img1)

img2 = cv2.imread("./resources/geetestplan.jpeg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, img2_white_black = cv2.threshold(img2_gray, 200, 255, cv2.IMREAD_GRAYSCALE)

cv2.imshow('img2_white_black',img2_white_black)
c = cv2.waitKey()

result_imag2,contours2=getContours(img2_white_black)
cv2.imshow('image2',result_imag2)
c = cv2.waitKey()



img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, img1_white_black = cv2.threshold(img1_gray, 200, 255, cv2.IMREAD_GRAYSCALE)
cv2.imshow('img1_white_black',img1_white_black)
c = cv2.waitKey()

result_imag1,contours1=getContours(img1_white_black,idx=2)
cv2.imshow('image1',result_imag1)
c = cv2.waitKey()

print("contours1 shape has size: ", len(contours1))
print("contours2 shape has size: ", len(contours2))

for i in range(len(contours1)):
    score,ct=1,None
    for k in range(len(contours2)):
        ret = cv2.matchShapes(contours1[i],contours2[k],1,0.0)
        if ret <score:
            score = ret
            ct = ret
    print("find c1 matchs c2  i: ", i, ' k: ',k)
    result_imag1 = np.empty((img1_white_black.shape[0],img1_white_black.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(result_imag1, contours1, i, (0,255,0), 3)
    cv2.imshow('result_imag1',result_imag1)
    c = cv2.waitKey()

    result_imag2 = np.empty((img2_white_black.shape[0],img2_white_black.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(result_imag2, contours2, i, (0,255,0), 3)
    cv2.imshow('result_imag2',result_imag2)
    c = cv2.waitKey()
# match shapes: https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html#:~:text=OpenCV%20comes%20with%20a%20function,on%20the%20hu%2Dmoment%20values.
