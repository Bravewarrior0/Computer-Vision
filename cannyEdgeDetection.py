import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/muhamed/cv404-2020-assignment-01-sbme404-2020-team05/images/Einsh.jpg',0)
edges = cv2.Canny(img,100,200)

cv2.imshow('image' , edges)
cv2.waitKey(0)