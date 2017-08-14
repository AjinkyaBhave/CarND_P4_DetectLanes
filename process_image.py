import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm

# Crop image from this height to remove car bonnet artifacts
img_height_crop = 670

# Source points are chosen to form a quadrangle on lane lines in the bottom half of image
top_left  = [570, 470]
top_right = [720, 470]
bottom_right = [1130, img_height_crop] # Originally 720
bottom_left  = [200, img_height_crop]  # Originally 720
src_pts = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)

# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image
bottom_left  = [320, 720]
bottom_right = [900, 720]
top_left  = [350, 1]
top_right = [875, 1]
dst_pts = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)

def sobel_thresh(img, orient='x', sobel_kernel=7, thresh=(20, 400)):
    # Convert to Y-channel
    img_y = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,0]
    # Take the gradient in x and y separately
    if orient == 'x':
        sobel = cv2.Sobel(img_y, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img_y, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print('Sobel orientation should be x or y')
    # Take absolute value of gradient
    sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    # Create a binary mask where mag thresholds are met
    sobel_bin = np.zeros_like(scaled_sobel)
    sobel_bin[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sobel_bin

def dir_thresh(img, sobel_kernel=5, thresh=(np.pi/4, np.pi/2)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    dir_bin = np.zeros_like(grad_dir)
    dir_bin[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # Return this mask as your binary_output image
    return dir_bin

def mag_thresh(img, sobel_kernel=5, thresh=(100, 200)):
    # Take Y-channel of image
    img_y = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,0]
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img_y, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_y, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    grad_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_grad_mag = np.uint8(255 * grad_mag / np.max(grad_mag))
    # Create a binary mask where mag thresholds are met
    mag_bin = np.zeros_like(scaled_grad_mag)
    mag_bin[(scaled_grad_mag >= thresh[0]) & (scaled_grad_mag <= thresh[1])] = 1
    return mag_bin

def l_thresh(img, img_bin):
    # Convert to HLS color space and separate the L channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    # Threshold color channel
    l_bin = np.zeros_like(l_channel)
    # Take lower portion of image for intensity normalisation
    img_height_lower = l_channel.shape[0]//3
    img_bin = img_bin[img_height_lower:, :]
    mu, sigma = norm.fit(l_channel[img_height_lower:, :][img_bin == 0].flatten())
    l_bin[l_channel > (mu + 1.5*sigma)] = 1
    return l_bin

def b_thresh(img, thresh=(140,255)):
    # Convert to LAB color space and separate the B channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
    b_channel = lab[:,:,2]
    # Threshold color channel
    b_bin = np.zeros_like(b_channel)
    b_bin[(b_channel >= thresh[0]) & (b_channel <= thresh[1])] = 1
    return b_bin

def threshold_image(img, visualise=False):
    # Process image based on gradient thresholds for edges
    sobel_bin = sobel_thresh(img)
    # Process image based on intensity thresholds for white lines
    l_bin = l_thresh(img, sobel_bin)
    # Process image based on saturation thresholds for yellow lines
    b_bin = b_thresh(img)

    # Combined thresholded binary
    img_bin = np.zeros_like(img[:,:,0])
    img_bin[(b_bin==1) | (l_bin==1)] = 1

    if visualise:
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image.
        # It might be beneficial to replace this channel with something else.
        #color_bin = np.dstack((np.zeros_like(mag_bin, dtype=np.uint8), s_bin, l_bin))
        # Plot the result
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Image', fontsize=40)
        ax2.imshow(sobel_bin, cmap='gray')
        ax2.set_title('Sobel Image', fontsize=40)
        ax3.imshow(l_bin, cmap='gray')
        ax3.set_title('L Image', fontsize=40)
        ax4.imshow(b_bin, cmap='gray')
        ax4.set_title('B Image', fontsize=40)
        ax5.imshow(img_bin, cmap='gray')
        ax5.set_title('Binary Result', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return img_bin

def undistort_image(img, mtx, dist, visualise=False):
    # Undistort image using camera intrinsic parameters
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    if visualise:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(img_undist, cmap='gray')
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    return img_undist

def view_road_top(img_bin, img=None, visualise=False):
# Create a top-view projection of binary image based on road plane assumption
    img_size = (img_bin.shape[1], img_bin.shape[0])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_top = cv2.warpPerspective(img_bin, M, img_size, flags=cv2.INTER_LINEAR)

    if visualise:
        img_vis = np.copy(img)
        img_top_vis = np.copy(img_top)
        # Plot original and projected image to check validity of transforrm
        #cv2.polylines(img_vis, np.array([src_pts],dtype=np.int32),True,(255,0,0), 5)
        #cv2.polylines(img_top_vis, np.array([dst_pts], dtype=np.int32),True,(0,0, 255), 5)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img_vis)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(img_top_vis, cmap='gray')
        ax2.set_title('Projected Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return img_top

def hist_image(img_gray, img_bin, visualise=False):
    # Take lower portion of image for intensity normalisation
    img_height_lower = 2 * img_gray.shape[0] // 3
    # Use only bottom part of image for histogram calculation
    img_bin = img_bin[img_height_lower:, :]
    # Convert to grayscale
    #img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,2]
    # Take a histogram of pixel intensity of the bottom third of the image
    hist, bins = np.histogram(img_gray[img_height_lower:, :][img_bin == 0], range(0, 256))
    max_idx = np.argmax(hist)
    avg_level = bins[max_idx-1]
    # best fit of data
    mu, sigma = norm.fit(img_gray[img_height_lower:, :][img_bin == 0].flatten())
    print('Mu: {}, Sigma {} Avg: {}'.format(mu, sigma, avg_level))
    img_hist = np.zeros_like(img_gray, dtype=np.uint8)
    img_hist[img_gray > (mu + 2*sigma)] = 1

    if visualise:
        # Plot histogram of the bottom half of the image
        plt.plot(bins[:-1], hist)
        plt.show()
        print('Avg. Level: ', avg_level)

    return img_hist