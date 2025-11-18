import sys
import math

import numpy as np
import cv2 

def get_coordination(coordination_contours, x, y, img):
    origin_coor = []
    for contour in coordination_contours:
        M = cv2.moments(contour) 
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        point_color = img[cy, cx]

        coor_color = {'x':[185, 120, 52], 'y':[134, 145, 33], 'o':[29, 45, 196]}

        for k in coor_color:
            if(sum([abs(int(coor_color[k][i]) -  int(point_color[i])) for i in range(3)])<=40):
                if k == 'x':
                    X = [cx, cy]
                elif k == 'y':
                    Y = [cx, cy]
                else:
                    O = [cx, cy]

    ex = [(X[i] - O[i])/100 for i in range(2)]
    ey = [(Y[i] - O[i])/100 for i in range(2)]
    A = np.column_stack((ex, ey))
    v = [x-O[0], y-O[1]]
    b = np.array(v)
    px, py = np.linalg.solve(A, b)
    return int(px), int(py)
    
def get_xaxis(coordination_contours,img):
    origin_coor = []
    for contour in coordination_contours:
        M = cv2.moments(contour) 
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        point_color = img[cy, cx]

        coor_color = {'x':[185, 120, 52], 'y':[134, 145, 33], 'o':[29, 45, 196]}

        for k in coor_color:
            if(sum([abs(int(coor_color[k][i]) -  int(point_color[i])) for i in range(3)])<=40):
                if k == 'x':
                    X = [cx, cy]
                elif k == 'y':
                    Y = [cx, cy]
                else:
                    O = [cx, cy]
    return X[0] - O[0], X[1] - O[1]

def calu_prob_point(cx, cy, x, y, angle_deg=50, ratio=0.6):
    theta = math.radians(angle_deg)
    x0 = x - cx
    y0 = y - cy
    xr = x0 * math.cos(theta) - y0 * math.sin(theta)
    yr = x0 * math.sin(theta) + y0 * math.cos(theta)
    return int(ratio*xr + cx), int(ratio*yr + cy)   

def calu_up_to_down(cx, cy, x1, y1, x2, y2):
    vector_o = [cx-x1, cy-y1]
    max_vx,max_vy = x2-x1,y2-y1
    vector = [x2-x1,y2-y1]
    cross = vector_o[0]*vector[1] - vector_o[1]*vector[0]
    if cross>=0:
        return True
    else:
        return False

def extract_info(image_path):
    img = cv2.imread(image_path)  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)

   
    
    kernel = np.ones((9,9), np.uint8)

    opening = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    blur = cv2.GaussianBlur(opening, (3,3), 0)

    edges = cv2.Canny(blur, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    target_contours = []
    coordination_contours = []
    for contour in contours:
        
        area = cv2.contourArea(contour)
        if(700<=area<=800):
            coordination_contours.append(contour)
        elif area > 800:
            target_contours.append(contour)
    
    ax_x,ax_y = get_xaxis(coordination_contours,img)
    vector_x = np.array([ax_x,ax_y])
    
    res = []
    
    for contour in target_contours:
            d = {}
            area = cv2.contourArea(contour)
            arc = cv2.arcLength(contour, True)
            rect = cv2.minAreaRect(contour)
            ratio = (rect[1][0]*rect[1][1])/area

            # square or circle or triangle or semicircle
            if(max(rect[1])/min(rect[1]) <= 1.4):
                if(area<3200):
                    x, y, w, h = cv2.boundingRect(contour)
                    ratio_2 = w*h/area
                    if 1.15<=ratio_2<=1.25:
                        d['label'] = 'semicircle'
                    elif 1.25<ratio_2<=1.35:
                        d['label'] = 'circle'
                    else:
                         d['label'] = 'square'
                else:
                    if(ratio <= 1.2):
                         d['label'] = 'square'
                    else:
                         d['label'] = 'triangle'

            # rectangle or triangle or semicircle
            else:
                if ratio <= 1.2:
                     d['label'] = 'rectangle'
                elif ratio < 1.45:
                     d['label'] = 'semicircle'
                else:
                     d['label'] = 'triangle'
            M = cv2.moments(contour) 
            
            # coordination
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # print("cx:"+str(cx))
            # print("cy:"+str(cy))
            # print(rect[0])

             
            # contour_img = img.copy()
            # cv2.drawContours(contour_img, contour, -1, (0,255,0), 2)
            # cv2.imshow("Contours", contour_img)
            # cv2.waitKey(0)

            d['X'],d['Y'] = get_coordination(coordination_contours, cx, cy, img)
            
            if d['label'] == 'circle':
                d['angle'] = 0
            else:
                
                epsilon = 0.01 * arc
                approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                max_dist = 0
                for point1 in approx:
                    x1, y1 = point1[0]
                    for point2 in approx:
                        x2, y2 = point2[0]
                        curr_dist = (x1-x2)**2 + (y1-y2)**2
                        if curr_dist > max_dist:
                            max_dist = curr_dist
                            if x1>x2:
                                x1,x2 = x2,x1
                                y1,y2 = y2,y1
                            max_vx,max_vy = x2-x1,y2-y1
                            max_x1,max_y1 = int(x1),int(y1)
                            max_x2,max_y2 = int(x2),int(y2)
                            
    
                    
                

                vector_o = np.array([max_vx,max_vy])
                
                
                cross = vector_o[0]*vector_x[1] - vector_o[1]*vector_x[0]
                dot   = vector_o[0]*vector_x[0] + vector_o[1]*vector_x[1]
                raw_angle = int(np.degrees(np.atan2(cross, dot)))
                if d['label'] == 'square':
                    d['angle'] = (raw_angle -45)%90
                elif d['label'] == 'rectangle':

                    prob_x,prob_y = calu_prob_point(cx,cy,max_x1,max_y1)
                    # max_img = img.copy()
                    # cv2.circle(max_img, (prob_x, prob_y), radius=3, color=(0, 0, 255), thickness=-1)
                    # cv2.circle(max_img, (max_x1, max_y1), radius=3, color=(0, 0, 255), thickness=-1)
                    # cv2.circle(max_img, (max_x2, max_y2), radius=3, color=(0, 0, 255), thickness=-1)
                    # cv2.imshow("max_img", max_img)
                    # cv2.waitKey(0)
                    if blur[prob_y,prob_x] == 0:
                        d['angle'] = (raw_angle - 25)
                    else:
                        d['angle'] = (raw_angle + 25)

                    
                elif d['label'] == 'semicircle':
                    d['angle'] = raw_angle
                    if not calu_up_to_down(cx, cy, max_x1, max_y1, max_x2,max_y2):
                        d['angle'] = raw_angle + 180

                elif d['label'] == 'triangle':
                    d['angle'] = raw_angle
                    if not calu_up_to_down(cx, cy, max_x1, max_y1, max_x2,max_y2):
                        d['angle'] = raw_angle + 180
            if d['angle'] > 180:
                d['angle'] = d['angle'] - 360
            res.append(d)
    return res

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("path")
    else:
        res = extract_info(sys.argv[1])
        print(len(res))
        for d in res:
            print(d)