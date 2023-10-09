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
    if  v<20:
        l=0
    elif s<20 and 20<=v<80 :
        l=math.floor(((v/100-0.2)*10))+1   
    elif s<20 and 80<=v<=100:
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
            H=6
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
    cv2.imwrite("cuts/cut1_"+name[-2]+"_"+name[-1], imgAddMask1)
    cv2.imwrite("cuts/cut2_"+name[-2]+"_"+name[-1], imgAddMask2)
    cv2.imwrite("cuts/cut3_"+name[-2]+"_"+name[-1], imgAddMask3)
    
    
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
    H1,H2,H3=Color_hist("cuts/cut1_"+name[-2]+"_"+name[-1],
                        "cuts/cut2_"+name[-2]+"_"+name[-1],
                        "cuts/cut3_"+name[-2]+"_"+name[-1])
    #H2_remake=[]
    #H3_remake=[]
    #for i in H2:
        #H2_remake.append(i+"2")
    #for i in H3:
        #H3_remake.append(i+"2")
    #H=H1+H2_remake+H3_remake    
    return H1,H2,H3






