import cv2
import numpy as np

#
def contours_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            #print('Distance: ',dist)
            if abs(dist) < 10 :
                return True
            elif i==row1-1 and j==row2-1:
                return False
def merge_contours(list_contours):
    single_contour = []
    for c in list_contours:
        single_contour.extend(c) 
    single_contour = np.array(single_contour)
    return single_contour

def combine_contours(contours):
    n = len(contours)
    ct_idx_to_merge={} # key is index, value is root picture current index can merge too
    for i in range(n):
        ct_idx_to_merge[i]=i # means each ct is independent
    for i in range(n):
        for j in range(i+1,n):
            if contours_close(contours[i],contours[j]):
                if i == ct_idx_to_merge[i]:
                    ct_idx_to_merge[j]=i
                else:
                    ct_idx_to_merge[j]=ct_idx_to_merge[i]
    
    re_org = {}
    for i in range(n):
        if i == ct_idx_to_merge[i]:
            re_org[i]=[i]
        else:
            if ct_idx_to_merge[i] not in re_org:
                ct_idx_to_merge[i]=[ct_idx_to_merge[i]]
            re_org[ct_idx_to_merge[i]].append(i)
    
    result = []
    for connected_contours in re_org.values():
        contours_after_merge = merge_contours([contours[i] for i in connected_contours])
        result.append(contours_after_merge)

    print("combined_contours size: ", len(result))
    return result
# https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
def getContours(img_white_black):
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
    # RETR_CCOMP: retrieves all of the contours and organizes them into a two-level hierarchy.
    #  At the top level, there are external boundaries of the components. 
    # At the second level, there are boundaries of the holes. 
    # If there is another contour inside a hole of a connected component, it is still put at the top level.

    # hierarchy + cv2.RETR_TREE can tell you parent-child relationship between contours:
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
    contours, hierarchy= cv2.findContours(img_white_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = combine_contours(contours)
    return contours

# retrun best matched contours and index of it in contours_list
def contourMatch(target_countours, contours_list):
    score,ct,idx=1,None,0
    for k in range(len(contours_list)):
        ret = cv2.matchShapes(target_countours,contours_list[k],1,0.0)
        if ret <score:
            score = ret
            ct = contours_list[k]
            idx=k
    return idx, ct
