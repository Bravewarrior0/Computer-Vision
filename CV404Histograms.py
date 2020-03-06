import numpy as np
import CV404Filters as myFilters
import cv2

def histogram(img):
    row, col = img.shape
    histo = {}
    for i in range(row):
        for j in range(col):
            histo[img[i, j]] = histo.get(img[i, j], 0)+1

    sortedHisto = sorted(histo.items())          
    return histo,sortedHisto

def equalization(img):
    equalized_img = img
    histo,sortedHisto = histogram(img)
    row = img.shape[0]
    col =img.shape[1]
    total_num = row*col 
    lvl =255
    pdf = {}
    cdf = {}
    roundoff={}
    equalized={}
    for i in range(len(sortedHisto)):
        pdf[sortedHisto[i][0]] = sortedHisto[i][1]/total_num
    for j in range(len(pdf)):
        if j ==0:
            cdf[sortedHisto[0][0]] = pdf[sortedHisto[0][0]]
        else:    
            cdf[sortedHisto[j][0]] =  cdf[sortedHisto[j-1][0]]  + pdf[sortedHisto[j][0]] 
    roundoff = cdf
    for key in roundoff:
        roundoff[key] = round(roundoff[key]*lvl)
        equalized[roundoff[key]] = equalized.get(roundoff[key],0)+ histo[key]

    for i in range(row):
        for j in range(col):
                equalized_img[i,j] = roundoff[equalized_img[i,j]]

    return equalized_img

# --------------thersholding----------------------------#
def threshold_global(gray_image , threshold =0.5 ):
    threshold = np.max( gray_image ) * threshold
    return 1 * ( gray_image > threshold )

def threshold_global_auto(gray_image):
    threshold = np.sum(gray_image)/(gray_image.size)
    return 1 * ( gray_image > threshold )

def threshold_local(img, filter_size = 11, filter_type ='gaussian'):
    '''
    filter_type can be 'gaussian', 'average', 'median'
    '''
    filter_type = filter_type.lower()
    thershold_img = None
    if(filter_type == 'average'):
        thershold_img  = myFilters.average_filter(img,filter_size)
    # elif(filter_type is 'median'):
    #     thershold_img = myFilters.median_filter(img, filter_size)
    else:
        thershold_img = myFilters.img_gaussian_filter(img, filter_size)
    return 1*(img > thershold_img)