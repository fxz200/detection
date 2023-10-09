#encoding = utf-8

import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import math
import matplotlib.pyplot as plt
def HSV_differentiate(HSV):
    h=HSV[0]*2
    s=HSV[1]*100/255
    v=HSV[2]*100/255
    if v>0 and v<=20:
        l=0
    elif 0<s<=20 and 20<=v<80 :
        l=math.floor(((v/100-0.2)*10))+1   
    elif 0<s<=20 and 20<v<=100:
        l=7
    else:
        if h>22 and h<=45:
            H=1
        elif h>45 and h<=70:
            H=2
        elif h>70 and h<=155:
            H=3
        elif h>155 and h<=186:
            H=4
        elif h>186 and h<=278:
            H=5
        elif h>278 or h<=330:
            H=0
        else:H=0
        if s>20 and s<=65:
            S=0
        else:S=1
        if v>20 and v<=70:
            V=0
        else:V=1
        l=4*H+2*S+V+8
    return l

def Centroid(img):
    img = cv2.imread(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,127,255,0)
    M = cv2.moments(thresh)
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    if cX>cY:
        radius=cY-1
    else:radius=cX-1
        
    cv2.circle(img, (cX, cY), 2, (255, 255, 255), -1)
    cv2.circle(img, (cX, cY), radius,(255, 255, 255), 0)
    cv2.circle(img, (cX, cY), radius//3,(255, 255, 255), 0)
    cv2.circle(img, (cX, cY), radius*2//3,(255, 255, 255), 0)
    print(cX,cY)
 
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    
def Radius(img):
    img = cv2.imread(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,127,255,0)
    M = cv2.moments(thresh)
  
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    if cX>cY:
        radius=cY-1
    else:radius=cX-1
    
    return radius

def Cut(img):
    name=img.split("/")
    
    img_=cv2.imread(img)
    img1=cv2.imread((img))
    img2=cv2.imread((img))
    img3=cv2.imread((img))
  
    r=Radius(img)

    MASK1=np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8) 
    cv2.circle(MASK1, (111,124), r, (255, 255, 255), -1) 
 
    MASK2=np.zeros((img2.shape[0], img2.shape[1]), dtype=np.uint8) 
    cv2.circle(MASK2, (111,124), r*2//3, (255, 255, 255), -1)


    MASK3=np.zeros((img3.shape[0], img3.shape[1]), dtype=np.uint8) 
    cv2.circle(MASK3, (111,124), r//3, (255, 255, 255), -1)
    mask2 = cv2.subtract(MASK2, MASK3)
    mask1 = cv2.subtract(MASK1, MASK2)
    
    imgAddMask1 = cv2.add(img1, np.zeros(np.shape(img1), dtype=np.uint8), mask=mask1)
    imgAddMask2 = cv2.add(img2, np.zeros(np.shape(img2), dtype=np.uint8), mask=mask2)
    imgAddMask3 = cv2.add(img3, np.zeros(np.shape(img3), dtype=np.uint8), mask=MASK3)
    cv2.imwrite("cut1_"+name[-2]+"_"+name[-1], imgAddMask1)
    cv2.imwrite("cut2_"+name[-2]+"_"+name[-1], imgAddMask2)
    cv2.imwrite("cut3_"+name[-2]+"_"+name[-1], imgAddMask3)
    
    
def Color_hist(cut1,cut2,cut3):
    img1 = cv2.imread(cut1)
    img2 = cv2.imread(cut2)
    img3 = cv2.imread(cut3)
    
    img1_=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    img2_=cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
    img3_=cv2.cvtColor(img3,cv2.COLOR_BGR2HSV)
    

    black=np.array([0,0,0])
    color_list_1_no_black=[]
    color_list_2_no_black=[]
    color_list_3_no_black=[]
    
    for i in img1_:
        for k in i:
            if k.any() == black.any():
                continue
            else:
                color_list_1_no_black.append(k)
                #print(k)
    for i in img2_:
        for k in i:
            if k.any() == black.any():
                continue
            else:
                color_list_2_no_black.append(k)
                #print(k)
    for i in img3_:
        for k in i:
            if k.any() == black.any():
                continue
            else:
                color_list_3_no_black.append(k)
                #print(k)    

    color_hist_1=[]
    for i in color_list_1_no_black:
        color_hist_1.append(HSV_differentiate(i))
    plt.subplot(131) 
    plt.hist(color_hist_1)

    color_hist_2=[]
    for i in color_list_2_no_black:
        color_hist_2.append(HSV_differentiate(i))
    plt.subplot(132) 
    plt.hist(color_hist_2)


    color_hist_3=[]
    for i in color_list_3_no_black:
        color_hist_3.append(HSV_differentiate(i))
    plt.subplot(133) 
    plt.hist(color_hist_3)


    return color_hist_1,color_hist_2,color_hist_3

def Compare_Hist(h1,h2):
    H1=plt.hist(h1,108,(0,108))
    H2=plt.hist(h2,108,(0,108))

    H1_array=H1[0]
    H1_array=H1_array[:,np.newaxis]
    H2_array=H2[0]
    H2_array=H2_array[:,np.newaxis]
    H1_array=np.float32(H1_array)
    H2_array=np.float32(H2_array)
    corr=(cv2.compareHist(H1_array,H2_array , method=0))#HISTCMP_CORREL
    dis=(cv2.compareHist(H1_array,H2_array , method=cv2.HISTCMP_BHATTACHARYYA))
    
    return corr ,dis


def To_Data(img):
    name=img.split("/")
    Cut(img)
    H1,H2,H3=Color_hist("cut1_"+name[-2]+"_"+name[-1],
                        "cut2_"+name[-2]+"_"+name[-1],
                        "cut3_"+name[-2]+"_"+name[-1])
    H2_remake=[]
    H3_remake=[]
    for i in H2:
        H2_remake.append(i*2)
    for i in H3:
        H3_remake.append(i*3)
    H=H1+H2_remake+H3_remake    
    return H

import os
import cv2
import time
import imutils
import numpy as np
sift = cv2.SIFT_create()


def parse_sift_output(target_path):
    """
    Return:
        kp: keypoint of hessian affine descriptor. location, orientation etc... OpenCV KeyPoint format. 
        des: 128d uint8 np array
    """
    
    # print(os.listdir("./sample"))
    kp = []
    des = []
    with open(target_path, "r") as f:
        lines = list(map(lambda x: x.strip(), f.readlines()))
        num_descriptor = int(lines[1])
        lines = lines[2:]
        for i in range(num_descriptor):
            # print(i, lines[i])
            val = lines[i].split(" ")
            x = float(val[0])
            y = float(val[1])
            a = float(val[2])
            b = float(val[3])
            c = float(val[4])
            # TODO: generate ellipse shaped key point
            # Refer: https://math.stackexchange.com/questions/1447730/drawing-ellipse-from-eigenvalue-eigenvector
            # Refer: http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/display_features.m
            # Refer: http://www.robots.ox.ac.uk/~vgg/research/affine/detectors.html
            key_point = cv2.KeyPoint(x, y, 1)
            sift_descriptor = np.array(list(map(lambda x: int(x), val[5:])), dtype=np.uint8)
            kp.append(key_point)
            des.append(sift_descriptor)
        
    
    return kp, np.array(des)


def resize(img):
    h, w, _ = img.shape
    if w > 500:
        img = imutils.resize(img, width=500)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def hessian_sift_extractor(path1, path2):
    os.system('./hesaff_c++/hesaff {}'.format(path1))
    os.system('./hesaff_c++/hesaff {}'.format(path2))

    kp1, des1 = parse_sift_output(path1 + '.hesaff.sift')
    kp2, des2 = parse_sift_output(path2 + '.hesaff.sift')
    des1 = des1.astype('float32')
    des2 = des2.astype('float32')
    return kp1, des1, kp2, des2

def sift_extractor(img1, img2):
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2


def show_match(path1, path2):

    img1 = cv2.imread(path1, 1)
    img2 = cv2.imread(path2, 1)
    img1_gray  = resize(img1.copy())

    img2_gray  = resize(img2.copy())
    
    kp1, des1, kp2, des2 = hessian_sift_extractor(path1, path2)
    # kp1, des1, kp2, des2 = sift_extractor(img1_gray, img2_gray)

    #### FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50) 


    begin = time.time()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    

    del_same = []
    mat = 0
    for i, (m, n) in enumerate(matches):
        # ratio test as per Lowe's paper
        if m.distance < 0.6 * n.distance:
            mat += 1
            matchesMask[i] = [1, 0]
            del_same.append(m.trainIdx)
    count = len(set(del_same))
    print('time�G{:.5f}, points�G{}'.format(time.time() - begin, count))


    draw_params = dict(matchColor = (0, 255, 0),
                       singlePointColor = (255, 0, 0),
                       matchesMask = matchesMask,
                       flags = 0)
    img3 = cv2.drawMatchesKnn(img1_gray, kp1, img2_gray, kp2, matches, None, **draw_params)
    # img3 = cv2.drawKeypoints(img1, kp1, img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

    cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('img',img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('test.jpg', img3)



