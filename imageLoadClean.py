import cv2

def rgba2rgb(rgba_img_numpy):
    print("RGBA_imag_numpy shape: ",rgba_img_numpy.shape)
    a=rgba_img_numpy[:,:,0:3]
    b=rgba_img_numpy[:,:,3]
    c=a + b[:,:,None]
    return c

def load2whiteblack(img_path):
    img = cv2.imread(cv2.samples.findFile(img_path), cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED for PNG file to count transparency layer
    if img_path.strip().endswith(".png"):
        img1 = rgba2rgb(img)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret, img1_white_black = cv2.threshold(img1_gray, 200, 255, cv2.IMREAD_GRAYSCALE)
        return img1_white_black

    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_white_black = cv2.threshold(img_gray, 200, 255, cv2.IMREAD_GRAYSCALE)
        return img_white_black
