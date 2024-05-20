import cv2
from skimage.io import imread
from scipy.ndimage import convolve, shift, label
from scipy import signal
from matplotlib import pyplot as plt
from ipywidgets import interactive
from IPython.display import display
import matplotlib as mpl
import numpy as np

mpl.rc('image', cmap='gray')  # tell matplotlib to use gray shades for grayscale images

test_im = np.array(imread("bench_images_cornerDet/b2.png", as_gray=True), dtype=float)  # image is floating point 0.0 to 1.0!
height, width = test_im.shape
test_im = cv2.resize(test_im, (60, 40), cv2.INTER_LINEAR)
print("Test image shape: ", test_im.shape)
print("test image tape: ", type(test_im))
test_im = test_im.astype(np.float32)
print("type of testim:", type(test_im))
plt.imshow(test_im)
plt.show()





def get_sobel_kernels():
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return kernel_x, kernel_y

    
def gaussian_filter(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def getcorners( img, k):

    # Get Sobel kernels
    sobel_kernel_x, sobel_kernel_y = get_sobel_kernels()
    # Compute gradients using Sobel operator
    grad_x = cv2.filter2D(img, ddepth=-1, kernel=sobel_kernel_x)
    grad_y = cv2.filter2D(img, ddepth=-1, kernel=sobel_kernel_y)
    #grad_y = signal.convolve2d(img, sobel_kernel_y, boundary="fill", mode="same")

    # Compute products of derivatives
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y
    Ixy = grad_x * grad_y

    # Apply Gaussian filter to the products of derivatives
    kernel_size = 3
    # Gaussian Kernel
    G = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]])/16
    sigma = 1
    Ixx = cv2.filter2D(Ixx, ddepth=-1, kernel=G)
    Iyy = cv2.filter2D(Iyy, ddepth=-1, kernel=G)
    Ixy = cv2.filter2D(Ixy, ddepth=-1, kernel=G)

    # Compute the Harris response
    det_M = Ixx * Iyy - Ixy * Ixy
    trace_M = Ixx + Iyy
    harris_response = det_M - k * (trace_M * trace_M)
    return harris_response
    # Threshold the Harris response to detect corners
    #harris_threshold = threshold * harris_response.max()
    #corners = harris_response > harris_threshold
    #
   #
    #return {"corners": corners, "harris_response": harris_response}

def get_harris_corners(image, k):
    R = getcorners(image, k)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(R > 1e-2))
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    return cv2.cornerSubPix(image, np.float32(centroids), (9,9), (-1,-1), criteria)



#test_im /= test_im.max()
corners = get_harris_corners(test_im, 0.04)

image_out = np.dstack((test_im , test_im, test_im))
print(type(corners))
for(x,y) in corners:
    x=np.round(x).astype(int)
    y = np.round(y).astype(int)
    cv2.circle(image_out, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
print("image is: ")
plt.imshow(image_out)
plt.show()