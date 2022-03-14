import cv2
from skimage import io as ski_io
import numpy as np

def rgba2rgb(rgba_img_numpy):
    print("RGBA_imag_numpy shape: ",rgba_img_numpy.shape)
    a=rgba_img_numpy[:,:,0:3]
    b=rgba_img_numpy[:,:,3]
    c=a + b[:,:,None]
    return c

def load2whiteblack(img_path):
    img=None
    if img_path.strip().startswith('http'):
        # ref: https://stackoverflow.com/questions/21061814/how-can-i-read-an-image-from-an-internet-url-in-python-cv2-scikit-image-and-mah
        img = ski_io.imread(img_path)
    else:
        img = cv2.imread(cv2.samples.findFile(img_path), cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED for PNG file to count transparency layer
    if img_path.strip().endswith(".png"):
        img = rgba2rgb(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_white_black = cv2.threshold(img_gray, 230, 255, cv2.IMREAD_GRAYSCALE)

    # 进行开运算操作
    # clean noise: https://zhuanlan.zhihu.com/p/36433309
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(img_white_black,cv2.MORPH_CLOSE,kernel)
    return opening

