import numpy as np
import cv2 as cv

def fill_result(result, fill_image):
    for i in range(fill_image.shape[0]):
        for j in range(fill_image.shape[1]):
            if(np.sum(result[i, j].astype(np.int32))==0):
                result[i,j] = fill_image[i,j]

def merge_image(image1_path, image2_path, save_path, MIN_MATCH_COUNT=0, mode=0):

    image1 = cv.imread(image1_path)          
    image2 = cv.imread(image2_path)          
    
    img1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    if mode == 1:
        image1,image2 = image2,image1
        img1,img2 = img2,img1
        
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
        if m.distance < 0.8*n.distance:
            good.append((m.trainIdx, m.queryIdx))
            

    
    if len(good) > 4:
        pts1 = np.float32([kp1[i].pt for(_,i) in good])
        pts2 = np.float32([kp2[i].pt for(i,_) in good])
        
        (H, status) = cv.findHomography(pts1, pts2, cv.RANSAC, 4.0)
        
        inlier_count = np.sum(status)
        if inlier_count < MIN_MATCH_COUNT:
            return False
        

        result_shape1 = image1.shape[1]+image2.shape[1]
        result_shape0 = image1.shape[0]+image2.shape[0]
        result = cv.warpPerspective(image1, H, (result_shape1, result_shape0))
        
        fill_result(result, image2)

        cv.imwrite(save_path, result)
        return True

def draw_compare(image1_path, image2_path, result, n):
    image1 = cv.imread(image1_path)          
    image2 = cv.imread(image2_path)     
    img1 = cv.imread(image1_path,cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread(image2_path,cv.IMREAD_GRAYSCALE) # trainImage
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf_norm = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches_norm = bf_norm.match(des1,des2)
    matches_norm = sorted(matches_norm, key = lambda x:x.distance)
    img3 = cv.drawMatches(image1,kp1,image2,kp2,matches_norm[:n],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   
    # Save the image
    cv.imwrite(result, img3)

if __name__=="__main__":
    draw_compare('foto1B.jpg', 'foto1A.jpg','result_demo.jpg',30)
    
    # merge_image('foto1A.jpg', 'foto1B.jpg', 'homoresult_sift.jpg')
    # draw_compare('anime.png', 'real_life.png','result_demo2.jpg',10)

    print(merge_image('real_life.jpg', 'anime.jpg', 'mergeresult_demo2_v.jpg',mode=1))
    print(merge_image('real_life.jpg', 'anime.jpg', 'mergeresult_demo2_h.jpg',mode=2))