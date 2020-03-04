import numpy as np 
import cv2 
import random


img = cv2.imread('/home/muhamed/cv404-2020-assignment-01-sbme404-2020-team05/images/girlWithScarf.png',0)
print(img.shape)
uniformNoise = np.zeros(img.shape, np.uint8)
# print(zeroMatrix)
row, col = img.shape

for x in range (row):
    for y in range (col):
        uniformNoise[x][y] = (random.uniform(0,255) + img[x][y]) / 2 

cv2.imshow('image',uniformNoise)
cv2.waitKey(0)