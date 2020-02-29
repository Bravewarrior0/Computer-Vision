def rgb2gray(rgb_image):
    # convert RGB img to grayScale img
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])