import numpy as np
import cv2 as cv
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('/home/muhamed/cv404-2020-assignment-01-sbme404-2020-team05/images/Einsh.jpg', 0)

img_float32 = np.float32(img)

dft = cv2.dft(np.float32(img_float32), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# cv.normalize(magnitude_spectrum, magnitude_spectrum, 0, 1, cv.NORM_MINMAX) # Transform the matrix with float values into aV
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)  # center

# Circular HPF mask, center circle is 0, remaining all ones

# mask = np.ones((rows, cols, 2), np.uint8)
# r = 300
# center = [crow, ccol]
# x, y = np.ogrid[:rows, :cols]
# mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
# mask[mask_area] = 1

# # create a mask first, center square is 1, remaining all zeros
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0


# apply mask and inverse DFT
fshift = dft_shift * mask

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


cv.normalize(img_back, img_back, 0, 1, cv.NORM_MINMAX) # Transform the matrix with float values into aV

cv2.imshow('high pass Filter',img_back )
cv2.waitKey(0)
