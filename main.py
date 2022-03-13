import cv2
import numpy as np

img1 = cv2.imread(cv2.samples.findFile("./resources/geetestObj3.png"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(cv2.samples.findFile("./resources/geetestplan.jpeg"), cv2.IMREAD_GRAYSCALE)

# Threshold both images first before using cv2.findContours
img1 = np.invert(img1)
ret, img2 = cv2.threshold(img2, 200, 255, 0)

if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
#-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
minHessian = 400
detector = cv2.SIFT_create(edgeThreshold=10)
keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
#-- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
#-- Filter matches using the Lowe's ratio test
ratio_thresh = 0.8
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
#-- Draw matches
img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#-- Show detected matches
print("Found matches size: ", len(img_matches))
cv2.imshow('Good Matches', img_matches)
cv2.waitKey()