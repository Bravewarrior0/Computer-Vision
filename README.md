# Computer Vision
This repo is the applying of basics conecpts that have been learnt in [SBE404B-2020](https://sbme-tutorials.github.io/2020/cv/) class of faculty of Engineering Cairo University SBE department.

## Objectives
* Filtration of noisy images using low pass filters such as: average, Gaussian, median.
* Edge detection using variety of masks such as: Sobel, Prewitt, and canny edge detectors.
* Histograms and equalization.
* Frequency domain filters.
* Hybrid images.
* Apply Hough transform for detecting parametric shapes like circles and lines.
* Apply Harris operator for detecting corners.
* Apply Active Contour Model for semi-supervised shape delineation.
* Apply Template Matching method using different similarity metrics.
* Apply SIFT and matching images with different rotations and scales.

## Computer Vision Functions 
1. Adding additive noise to the image.
   - Uniform, Gaussian and salt & pepper noise.
2. Filtering the noisy image using the following
   - low pass filters,Average, Gaussian and median filters.
3. Detecting edges in the image using the following masks
   - Sobel, Roberts , Prewitt and canny edge detectors.
4. Draw histogram and distribution curve.
5. Equalize the image.
6. Normalize the image.
7. Local and global thresholding.
8. Transformation from color image to gray scale image and plot of R, G, and B histograms with its distribution function (cumulative curve that you use it for mapping and histogram equalization).
9. Frequency domain filters (high pass and low pass).
10. Hybrid images.
11. Detecting lines and circles located in images using Hough transform.
12. Detecting corners using Harris operator.
13. Initializing the contour for a given object and evolve the Active Contour Model (snake) using the greedy algorithm.
14. Matching the image set features using: 
   1. Correlation 
   2. Zero-mean correlation
   3. Sum of squared differences (SSD) 
   4. Normalized cross correlations. 
15. Generating feature descriptors using scale invariant features (SIFT).

### Files
1. `CV404Filters.py`: include implementation of filtration functions (1-3).
2. `CV404Histograms.py`: include implementation of histogram (4-8).
3. `CV404Frequency.py`: include implementation of frequency domain (9-10).
4. `CV404Hough.py`: include implementation of Hough transform for lines and circles (11).
5. `CV404Harris.py`: include implementation of Harris operator for corners detection (12)
6. `CV404ActiveContour.py`: include implementation of the Active Contour Model  (13)
7. `CV404Template.py`: include the implementation of template matching functions (14).
8. `CV404SIFT.py`: include the implementation of SIFT technique (15).

## Test
To try the project run gui.py

