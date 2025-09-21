import numpy as np
import cv2 as cv


if __name__=="__main__":
    image1 = cv.imread('foto1A.jpg')          
    image2 = cv.imread('foto1B.jpg')          
    
    img1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    # Initiate SIFT detector
    sift = cv.SIFT_create()
     
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append((m.trainIdx, m.queryIdx))
    
    if len(good) > 4:
        pts1 = np.float32([kp1[i].pt for(_,i) in good])
        pts2 = np.float32([kp2[i].pt for(i,_) in good])
        
        (H, status) = cv.findHomography(pts1, pts2, cv.RANSAC, 4.0)
        
        result = cv.warpPerspective(image1, H, (image1.shape[1]+image1.shape[1], image1.shape[0]))
        
        cv.imwrite('result_merge1.jpg', result)
        
        result[0:image2.shape[0], 0:image2.shape[1]] = image2
        cv.imwrite('homoresult_sift.jpg', result)