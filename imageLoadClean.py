import cv2
import numpy as np
from PIL import Image
import os.path 


def rgba2rgb(rgba_img_numpy):
    print("RGBA_imag_numpy shape: ",rgba_img_numpy.shape)
    a=rgba_img_numpy[:,:,0:3]
    b=rgba_img_numpy[:,:,3]
    c=a + b[:,:,None]
    return c

