import numpy as np
import cv2
import matplotlib.pyplot as plt

# Source points are chosen to form a quadrangle on lane lines in the bottom half of image
top_left  = [570, 470]
top_right = [720, 470]
bottom_right = [1130, 720]
bottom_left  = [200, 720]
src_pts = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)

# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image
bottom_left  = [320, 720]
bottom_right = [920, 720]
top_left  = [322, 1]
top_right = [918, 1]
dst_pts = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)

def sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(100, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print('Sobel orientation should be x or y')
        grad_binary = np.copy(img)
    # Take absolute value of gradient
    sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    # Create a binary mask where mag thresholds are met
    sobel_bin = np.zeros_like(scaled_sobel)
    sobel_bin[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sobel_bin

def mag_thresh(img, sobel_kernel=3, thresh=(120, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    grad_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_grad_mag = np.uint8(255 * grad_mag / np.max(grad_mag))
    # Create a binary mask where mag thresholds are met
    mag_bin = np.zeros_like(scaled_grad_mag)
    mag_bin[(scaled_grad_mag >= thresh[0]) & (scaled_grad_mag <= thresh[1])] = 1
    return mag_bin

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

def s_thresh(img, thresh=(170,255)):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    # Threshold color channel
    s_bin = np.zeros_like(s_channel)
    s_bin[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_bin

def v_thresh(img, thresh=(170,255)):
    # Convert to HSV color space and separate the S channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    v_channel = hsv[:, :, 2]
    # Threshold color channel
    v_bin = np.zeros_like(v_channel)
    v_bin[(v_channel >= thresh[0]) & (v_channel <= thresh[1])] = 1
    return v_bin

def threshold_image(img, visualise=False):
    # Process image based on gradient and color thresholds
    # Choose a Sobel kernel size
    ksize = 7  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    #gradx_bin = sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(100, 255))
    #grady_bin = sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(100, 255))
    #dir_bin = dir_thresh(img, sobel_kernel=ksize, thresh=(np.pi / 4, np.pi / 5))
    mag_bin = mag_thresh(img, sobel_kernel=ksize, thresh=(150, 255))
    s_bin = np.zeros_like(mag_bin)
    v_bin = v_thresh(img)

    # Combined thresholded binary
    img_bin = np.zeros_like(s_bin)
    img_bin[(mag_bin == 1) | (v_bin == 1)] = 1

    # Use only bottom part of image for histogram calculation
    img_bin = img_bin[2 * img.shape[0] // 3:, :]
    # Convert to grayscale
    #img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,2]
    # Take a histogram of pixel intensity of the bottom third of the image
    hist, bins = np.histogram(img_gray[2 * img.shape[0] // 3:, :][img_bin == 0], range(0, 256))
    max_idx = np.argmax(hist)
    avg_level = bins[max_idx]
    img_hist = np.zeros_like(img_gray, dtype=np.uint8)
    img_hist[img_gray > (avg_level + 40)] = 1

    if visualise:
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image.
        # It might be beneficial to replace this channel with something else.
        color_bin = np.dstack((mag_bin, s_bin, v_bin))
        # Plot the result
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img_gray, cmap='gray')
        ax1.set_title('Original Image', fontsize=40)
        ax2.imshow(img_hist, cmap='gray')
        ax2.set_title('Color Result', fontsize=40)
        ax3.imshow(img_bin, cmap='gray')
        ax3.set_title('Binary Result', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return img_hist

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
        img_top_vis = np.copy(img_top)
        # Plot original and projected image to check validity of transforrm
        cv2.polylines(img, np.array([src_pts],dtype=np.int32),True,(255,0,0), 5)
        cv2.polylines(img_top_vis, np.array([dst_pts], dtype=np.int32),True,(0,0,255), 5)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(img_top_vis, cmap='gray')
        ax2.set_title('Projected Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    return img_top

def hist_image(img, visualise=False):
    # Calculate thresholded gradient magnitude image
    mag_bin = mag_thresh(img, sobel_kernel=7, thresh=(150, 255))
    # Calculate thresholded s-channel image
    s_bin = s_thresh(img)
    # Calculate thresholded s-channel image
    l_bin = l_thresh(img)
    # Combined thresholded binary
    img_bin = np.zeros_like(s_bin)
    img_bin[(mag_bin == 1) | (s_bin == 1) | (l_bin == 1)] = 1
    # Use only bottom part of image for histogram calculation
    img_bin = img_bin[2*img.shape[0]//3:, :]
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take a histogram of pixel intensity of the bottom third of the image
    hist, bins = np.histogram(img_gray[2*img.shape[0]//3:,:][img_bin==0], range(0,256))
    max_idx = np.argmax(hist)
    avg_level = bins[max_idx]
    img_hist = np.zeros_like(img_gray, dtype=np.uint8)
    img_hist[img_gray > (avg_level + 50)] = 1

    if visualise:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(l_bin, cmap='gray')
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(img_hist, cmap='gray')
        ax2.set_title('Projected Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        print(avg_level)

    return img_hist