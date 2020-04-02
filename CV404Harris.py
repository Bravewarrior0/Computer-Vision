import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import CV404Filters as myFilters

def harris_response(img, w=3,k= 0.05, gaussian_size =3):
    '''
    w : window size

    '''
    img_gray = myFilters.rgb2gray(img)
    img_gray = myFilters.img_gaussian_filter(img_gray,gaussian_size)
    Ix = myFilters.sobel_h(img_gray)
    Iy = myFilters.sobel_v(img_gray)
    Ixx = np.multiply(Ix,Ix)
    Iyy = np.multiply(Iy,Iy)
    Ixy = np.multiply(Ix,Iy)

    Ixx_hat = myFilters.img_gaussian_filter(Ixx,w)
    Iyy_hat = myFilters.img_gaussian_filter(Iyy,w)
    Ixy_hat = myFilters.img_gaussian_filter(Ixy,w)

    detM = np.multiply(Ixx_hat,Iyy_hat) - np.multiply(Ixy_hat,Ixy_hat) 
    traceM = Ixx_hat + Iyy_hat
    R = detM - k * np.multiply(traceM,traceM) # Harris response
    return R

def get_corners(img,w=3, k=0.05,thershold = 0.0001):
    R = harris_response(img,w,k)
    R/=np.max(R)
    img_copy = np.copy(img)
    corners = R > thershold
    # imgplot = plt.imshow(corners ,cmap=plt.get_cmap('gray'))
    # plt.show()
    if(len(img.shape) == 2):
        img_copy [corners] = 1
    elif(len(img.shape)==3):
        if(img.shape[2]==3):
            img_copy [corners] = [255,0,0]
        elif(img.shape[2]==4):
            img_copy [corners] = [255,0,0,255]
    return img_copy

# img_c = mpimg.imread('images\chess_board.jpg')
# # img_c = mpimg.imread('images\\shapes.jpg')
# out = get_corners(img_c)

# imgplot = plt.imshow(out,cmap=plt.get_cmap('gray'))
# plt.show()