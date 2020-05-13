from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from CV404Filters import sobel , rgb2gray
from Canny import canny
import numpy as np
import cv2
from collections import defaultdict

def hough_line(img):
    #Get image dimensions
    Y, X = np.nonzero(img)
    #Max diatance is diagonal one 

    Maxdist = int(np.round(np.sqrt(img.shape[0]**2 + img.shape[1]** 2)))

    # 1. initialize parameter space rs, thetas
    # Theta in range from -90 to 90 degrees
    # thetas = np.deg2rad(np.arange(-90, 90))
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 180)
    #Range of radius
    rhos = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    
    #2. Create accumulator array and initialize to zero
    accumulator = np.zeros((2 * Maxdist, len(thetas)))

    #3. Loop for each edge pixel 
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        #4. Loop for each theta
        # Map edge pixel to hough space
        for k in range(len(thetas)):
            #5. calculate $\rho$
            # Calculate space parameter
            rho = x*np.cos(thetas[k]) + y * np.sin(thetas[k]) 

            #6. Increment accumulator at rho, theta
            # Update the accumulator
            # map r to its idx 0 : 2*max
            accumulator[int(rho)+ Maxdist,k] += 1

    return accumulator, thetas, rhos

def extract_lines( accumulator , thetas , rhos , threshold = 0.5) :
    theta_peak =[]
    rho_peak = []
    # threshold = np.quantile( accumulator , threshold )
    threshold = threshold * np.max(accumulator)
    acc2 = np.zeros(accumulator.shape)
    acc=[]
    for rho_idx in range(accumulator.shape[0]) :
        for theta_idx in range(accumulator.shape[1]) :
            if accumulator[rho_idx, theta_idx] > threshold :
                theta_peak.append(thetas[theta_idx])
                rho_peak.append(rhos[rho_idx])
                acc2[rho_idx,theta_idx] = accumulator[rho_idx, theta_idx]
    theta_peak = np.array(theta_peak)
    rho_peak = np.array(rho_peak)
    return acc2, theta_peak, rho_peak

def get_hough_lines(img, threshold = 0.3):
    gray = rgb2gray(img)
    edge = canny(gray)
    accs, ths, rs  = hough_line(edge)
    
    h,w = np.array(gray.shape)/100
    fig = plt.figure(figsize=(w,h))
    ax = fig.add_axes([0,0,1,1])

    ax.imshow(img,cmap='gray')
    ax.autoscale(False)
    ax.axis('off')
    origin = np.array((0, img.shape[1]))
    for _, angle, dist in zip(*extract_lines(accs, ths, rs,threshold)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax.plot(origin, (y0, y1), '-r')

    fig.canvas.draw()
    out = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close('all') 
    return out 

from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from Canny import canny_edge_detector
from collections import defaultdict

def hough_circles(src, rmin =20, rmax = 25,threshold =0.2, steps = 100):
    input_image = Image.open(src)
    output_image = Image.new("RGB", input_image.size)
    output_image.paste(input_image)
    draw_result = ImageDraw.Draw(output_image)
    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    for x, y in canny_edge_detector(input_image):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            circles.append((x, y, r))

    for x, y, r in circles:
        draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))

    # Save output image
    output_image.save("hough.png")
    return mpimg.imread('hough.png')