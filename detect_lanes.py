import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt

# Define a class to store the attributes of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Number of iterations for filtering calculations
        self.n_fit = 5
        # x values of the last n_fit of the line
        self.recentx = deque(maxlen=self.n_fit)
        # x values of the fitted line averaged over the last n_fit iterations
        self.avgx = None
        # polynomial coefficients of the last n_fit of the line
        self.recent_fit = deque(maxlen=self.n_fit)
        # polynomial coefficients averaged over the last n iterations
        self.avg_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # difference in fit coefficients between last and new fits
        self.diffs = np.zeros(self.fit_degree+1, dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # Degree of polynomial for line fit
        self.fit_degree = 2
        # radius of curvature of the line in metres
        self.radius_of_curvature = None
        # distance of vehicle center from the centre of lane in meters
        self.line_base_pos = None

# Objects containing attributes for left and right ego-lane lines
left_line  = Line()
right_line = Line()

# Set the width of the windows +/- margin
margin = 100
# Metres per pixel in x dimension
xm_per_pix = 3.7/700
# Metres per pixel in y dimension
ym_per_pix = 30/720

def sliding_window(img_bin, nonzerox, nonzeroy, visualise=False):
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_line_idx = []
    right_line_idx = []

    # Create an output image to draw on and  visualize the result
    # Image type needs to be uint8 to enable drawing of rectangles and points using cv2
    img_out = (np.dstack((img_bin, img_bin, img_bin)) * 255).astype(np.uint8)

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img_bin[img_bin.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Choose the number of sliding windows
    n_win = 9
    # Set height of windows
    win_height = np.int(img_bin.shape[0] / n_win)
    # Step through the windows one by one
    for win in range(n_win):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_bin.shape[0] - (win + 1) * win_height
        win_y_high = img_bin.shape[0] - win * win_height
        win_leftx_low = leftx_current - margin
        win_leftx_high = leftx_current + margin
        win_rightx_low = rightx_current - margin
        win_rightx_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_leftx_low) & (
                           nonzerox < win_leftx_high)).nonzero()[0]
        good_right_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_rightx_low) & (
                            nonzerox < win_rightx_high)).nonzero()[0]
        # Append these indices to the lists
        left_line_idx.append(good_left_idx)
        right_line_idx.append(good_right_idx)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_idx) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_idx]))
        if len(good_right_idx) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_idx]))
        if visualise:
            # Draw the windows on the visualization image
            cv2.rectangle(img_out, (win_leftx_low, win_y_low), (win_leftx_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(img_out, (win_rightx_low, win_y_low), (win_rightx_high, win_y_high), (0, 255, 0), 2)

    # Concatenate the arrays of indices
    left_line_idx = np.concatenate(left_line_idx)
    right_line_idx = np.concatenate(right_line_idx)

    if visualise:
        # Plot histogram of the bottom half of the image
        plt.plot(np.arange(0, img_bin.shape[1]), histogram)
        plt.show()
        # Draw left and right lines pixels on output image
        img_out[nonzeroy[left_line_idx], nonzerox[left_line_idx]] = [255, 0, 0]
        img_out[nonzeroy[right_line_idx], nonzerox[right_line_idx]] = [0, 0, 255]

    return left_line_idx, right_line_idx, img_out

def fit_lane(img_bin, visualise=False):
    # img_bin is the projected thresholded binary image from camera
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Start sliding window search if either left or right lines were not detected in previous frame
    if (not left_line.detected) or (not right_line.detected):
        left_line_idx, right_line_idx, img_out = sliding_window(img_bin, nonzerox, nonzeroy, visualise=visualise)
    else:
    # Otherwise start focused search around most recent left and right lines detected
        left_line_idx = ((nonzerox > (left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy + left_line.current_fit[2] - margin)) & (
                          nonzerox < (left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy + left_line.current_fit[2] + margin)))
        right_line_idx = ((nonzerox > (right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy + right_line.current_fit[2] - margin)) & (
                           nonzerox < (right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy + right_line.current_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_line_idx]
    lefty = nonzeroy[left_line_idx]
    rightx = nonzerox[right_line_idx]
    righty = nonzeroy[right_line_idx]

    # If pixels of left line not detected in current image
    if (leftx.size == 0) or (lefty.size == 0):
        # Use previous best fit as current fit
        left_line.detected = False
    else:
        # Left line pixels are detected accurately
        left_line.detected = True
        # Fit a second order polynomial to detected left line pixels
        left_fit = np.polyfit(lefty, leftx, left_line.fit_degree)
        ## Calculate goodness of fit here...
        ##
        # Calculate left line radius
        left_rad, _ = find_curvature(img_bin, 'left')
        # Calculate best fit to line
        if len(left_line.recent_fit) == left_line.n_fit:
            left_line.avg_fit = np.sum(np.asarray(left_line.recent_fit), axis=0)/left_line.n_fit
            left_line.recent_fit.pop(0)
        # Add current fit to end of list
        left_line.recent_fit.append(left_fit)

        # Save attributes of left lane object for current image
        left_line.allx = leftx
        left_line.ally = lefty
        left_line.diffs = left_line.current_fit - left_fit
        left_line.current_fit = left_fit
        left_line.radius_of_curvature = left_rad

    # If pixels of right line not detected in current image
    if (rightx.size == 0) or (righty.size == 0):
        # Use previous best fit as current fit
        right_line.detected = False

    else:
        # Right line pixels are detected accurately
        right_line.detected = True
        # Fit a second order polynomial to detected right line pixels
        right_fit = np.polyfit(righty, rightx, right_line.fit_degree)
        # Calculate left line radius
        _, right_rad = find_curvature(img_bin, 'right')
        # Save attributes of right line object for current image
        right_line.allx = rightx
        right_line.ally = righty
        right_line.diffs = right_line.current_fit - right_fit
        right_line.current_fit = right_fit
        right_line.radius_of_curvature = right_rad

    if visualise:
        plot_lanes(img_out, left_line.current_fit, right_line.current_fit)

    return left_line.current_fit, right_line.current_fit

def find_curvature(img_bin, lane_line):
    # Y-coordinates to calculate radius over
    y_fit = np.linspace(0, img_bin.shape[0]-1, img_bin.shape[0])
    # Y-coordinate of point to calculate the curvature at
    y_eval = int(img_bin.shape[0]/2)
    # Fit new polynomials to x,y in world space
    left_rad  = -1
    right_rad = -1
    if lane_line == 'left':
        left_fit_wc  = np.polyfit(y_fit * ym_per_pix, left_line.allx * xm_per_pix, left_line.fit_degree)
        # Calculate the new radii of curvature
        left_rad = ((1 + (2 * left_fit_wc[0] * y_eval * ym_per_pix + left_fit_wc[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * left_fit_wc[0])
    if lane_line == 'right':
        right_fit_wc = np.polyfit(y_fit * ym_per_pix, right_line.allx * xm_per_pix, right_line.fit_degree)
        right_rad = ((1 + (2 * right_fit_wc[0] * y_eval * ym_per_pix + right_fit_wc[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_wc[0])
    return left_rad, right_rad

def plot_lanes(img_out, left_fit, right_fit):
    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(img_out)

    # Generate x and y values for plotting
    y_fit = np.linspace(0, img_out.shape[0]-1, img_out.shape[0])
    leftx_fit = left_fit[0] * y_fit ** 2 + left_fit[1] * y_fit + left_fit[2]
    rightx_fit = right_fit[0] * y_fit ** 2 + right_fit[1] * y_fit + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([leftx_fit - margin, y_fit]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([leftx_fit + margin, y_fit])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([rightx_fit - margin, y_fit]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rightx_fit + margin, y_fit])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(img_out, 1, window_img, 0.3, 0)
    plt.imshow(result)
    # Draw lane lines based on current fit
    plt.plot(leftx_fit, y_fit, color='yellow')
    plt.plot(rightx_fit, y_fit, color='yellow')
    plt.xlim(0, img_out.shape[1])
    plt.ylim(img_out.shape[0], 0)
    plt.show()
