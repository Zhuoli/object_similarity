import cv2
import numpy as np

def feature_match(img1, img2):
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    # https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html
    detector = cv2.SIFT_create(edgeThreshold=50)
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