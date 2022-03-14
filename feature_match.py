import cv2
import numpy as np

# https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
def get_point_maps(kp1, kp2, matches):
    # Initialize lists
    list_kp1 = []
    list_kp2 = []

    # For each match...
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))
    return list_kp1, list_kp1

def feature_match(img1, img2, debug=False):
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
    kp1_list,kp2_list = get_point_maps(keypoints1,keypoints2, good_matches)
    print("Found good match points size: ", len(kp1_list))
    if debug:
        #-- Draw matches
        img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #-- Show detected matches
        print("Found matches size: ", len(img_matches))
        cv2.imshow('Good Matches', img_matches)
        cv2.waitKey()
    return kp1_list,kp2_list

# https://stackoverflow.com/questions/50670326/how-to-check-if-point-is-placed-inside-contour
def cal_best_match(points, contours):
    best_matched_contour_idx = -1
    best_matched_contour = None
    for p_idx in range(len(points)):
        for c_idx in range(len(contours)):
            dist = cv2.pointPolygonTest(contours[c_idx],points[p_idx],True)