import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt
from process_image import img_height_crop

# Video frames per second
video_FPS = 25
# Maximum number of missed frames allowed
max_missed_frames = int(np.ceil(0.1*video_FPS))
# Empirical minimum width of lanes in metre
min_lane_width = 3.7/2
# Minimum radius of curvature of U.S. highway lanes in metre
min_lane_radius = 200
# Ratio between current and previous radius of curvature
max_radius_ratio = 100
# Weight of current fit to average fit to keep for current calculation
fit_gain = 0.90
# Set the width of the windows +/- margin
win_margin = 50
# Set width of histogram search
hist_margin = 320
# Minimum pixels needed to accept line detected
min_pixels_line = 5
# Metres per pixel in x dimension
xm_per_pix = 3.7/700
# Metres per pixel in y dimension
ym_per_pix = 30/720
# Parameters of vehicle camera images
img_height = 720
img_width = 1280
img_channels = 3

# Define a class to store the attributes of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Degree of polynomial for line fit
        self.fit_degree = 2
        # Number of frames to calculate fit over
        self.n_fit = 5
        # Number of continuous frames with missing line
        self.n_missed_frames  = max_missed_frames
        # x values of the last n_fit of the line
        self.recentx = deque(maxlen=self.n_fit)
        # x values of the fitted line averaged over the last n_fit iterations
        self.avgx = None
        # polynomial coefficients of the last n_fit of the line
        self.recent_fit = deque(maxlen=self.n_fit)
        # polynomial coefficients averaged over the last n iterations
        self.avg_fit = np.array([0.0,0.0,0.0], dtype=np.float32)
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # difference in fit coefficients between last and new fits
        self.diffs = np.zeros(self.fit_degree+1, dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # radius of curvature of the line in metres
        self.radius = min_lane_radius
        # distance of vehicle center from the centre of lane in meters
        self.centre_offset = 0

# Objects containing attributes for left and right ego-lane lines
left_line  = Line()
right_line = Line()

def sliding_window(img_bin, img_out, nonzerox, nonzeroy, visualise=False):
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_line_idx = []
    right_line_idx = []

    # Take a histogram of the bottom half of the image
    hist_img = np.sum(img_bin[img_bin.shape[0]//2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = hist_img.shape[0]//2
    # Check for left and right lane peaks in a pre-defined margin around the midpoint of the image
    # This helps to avoid spurious detections at the image boundaries from road edges, other cars, fences, and bright dust surfaces.
    leftx_base = np.argmax(hist_img[midpoint-hist_margin:midpoint]) + midpoint-hist_margin
    rightx_base = np.argmax(hist_img[midpoint:midpoint+hist_margin]) + midpoint
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Choose the number of sliding windows
    n_win = 9
    # Maximum distance from bottom of image to search in
    win_search_height = img_bin.shape[0]
    # Set height of windows
    win_height = np.int(win_search_height/n_win)

    # Step through the windows one by one
    for win in range(n_win):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_bin.shape[0] - (win + 1) * win_height
        win_y_high = img_bin.shape[0] - win * win_height
        win_leftx_low = leftx_current - win_margin
        win_leftx_high = leftx_current + win_margin
        win_rightx_low = rightx_current - win_margin
        win_rightx_high = rightx_current + win_margin
        # Identify the nonzero pixels in x and y within the window
        good_left_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_leftx_low) & (
                           nonzerox < win_leftx_high)).nonzero()[0]
        good_right_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_rightx_low) & (
                            nonzerox < win_rightx_high)).nonzero()[0]

        # Detected more than minpix pixels in window
        if len(good_left_idx) > minpix:
            # Recenter next window on their mean position
            leftx_current = np.int(np.mean(nonzerox[good_left_idx]))
            # Append these indices to the lists
            left_line_idx.append(good_left_idx)

        # Detected more than minpix pixels in window
        if len(good_right_idx) > minpix:
            # Recenter next window on their mean position
            rightx_current = np.int(np.mean(nonzerox[good_right_idx]))
            # Append these indices to the lists
            right_line_idx.append(good_right_idx)

        if visualise:
            # Draw the windows on the visualization image
            cv2.rectangle(img_out, (win_leftx_low, win_y_low), (win_leftx_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(img_out, (win_rightx_low, win_y_low), (win_rightx_high, win_y_high), (0, 255, 0), 2)

    # Concatenate the arrays of indices
    if len(left_line_idx)!=0:
        left_line_idx = np.concatenate(left_line_idx)
    if len(right_line_idx)!=0:
        right_line_idx = np.concatenate(right_line_idx)

    if visualise:
        # Plot histogram of the bottom half of the image
        plt.plot(np.arange(0, img_bin.shape[1]), hist_img)
        plt.show()
        # Draw left and right lines pixels on output image
        if len(left_line_idx) != 0:
            img_out[nonzeroy[left_line_idx], nonzerox[left_line_idx]] = [255, 0, 0]
        if len(right_line_idx) != 0:
            img_out[nonzeroy[right_line_idx], nonzerox[right_line_idx]] = [0, 0, 255]

    return left_line_idx, right_line_idx

def fit_lane(img_bin, visualise=False):
    # img_bin is the projected thresholded binary image from camera
    # Create an output image to draw on and  visualize the result
    # Image type needs to be uint8 to enable drawing of rectangles and points using cv2
    img_out = (np.dstack((img_bin, img_bin, img_bin)) * 255).astype(np.uint8)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Indicate which line to search for based on correctly detected pixels and past frames
    search_left_line  = not left_line.detected and (left_line.n_missed_frames >= max_missed_frames)
    search_right_line = not right_line.detected and (right_line.n_missed_frames >= max_missed_frames)
    # Start sliding window search if either left or right lines were not detected in previous n_missed_frames frames
    if (search_left_line or search_right_line ) :
        left_line_idx, right_line_idx = sliding_window(img_bin, img_out, nonzerox, nonzeroy, visualise=visualise)
    else:
        # Otherwise start focused search around most recent left and right lines detected
        left_line_idx = ((nonzerox > (left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy + left_line.current_fit[2] - win_margin)) & (
                          nonzerox < (left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy + left_line.current_fit[2] + win_margin)))
        right_line_idx = ((nonzerox > (right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy + right_line.current_fit[2] - win_margin)) & (
                           nonzerox < (right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy + right_line.current_fit[2] + win_margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_line_idx]
    lefty = nonzeroy[left_line_idx]
    rightx = nonzerox[right_line_idx]
    righty = nonzeroy[right_line_idx]

    # Check if lines are correct based on previous frames
    left_fit, right_fit, left_rad, right_rad = check_lane_fit(lefty, leftx, righty, rightx)

    # Save left and right line parameters if correctly detected
    if left_line.detected == False:
        # Increment missed frames counter
        left_line.n_missed_frames += 1
    else:
        # Left line pixels are detected accurately in current image
        # Reset missed frames counter
        left_line.n_missed_frames = 0
        # Keep at most n_fit past fits
        if len(left_line.recent_fit) == left_line.n_fit:
            left_line.recent_fit.popleft()
        # Add current fit to end of list
        left_line.recent_fit.append(left_fit)
        # Save attributes of left lane object for current image
        left_line.allx = leftx
        left_line.ally = lefty
        left_line.diffs = np.abs(left_line.current_fit - left_fit)
        left_line.current_fit = np.asarray(left_fit)
        left_line.radius = left_rad
        # Calculate average fit to line
        left_line.avg_fit = np.sum(np.asarray(left_line.recent_fit), axis=0)/left_line.n_fit

    if right_line.detected == False:
        # Increment missed frames counter
        right_line.n_missed_frames += 1
    else:
        # Right line pixels are detected accurately
        # Reset missed frames counter
        right_line.n_missed_frames = 0
        # Keep at most n_fit past fits
        if len(right_line.recent_fit) == right_line.n_fit:
            right_line.recent_fit.popleft()
        # Add current fit to end of list
        right_line.recent_fit.append(right_fit)
        # Save attributes of right line object for current image
        right_line.allx = rightx
        right_line.ally = righty
        right_line.diffs = right_line.current_fit - right_fit
        right_line.current_fit = np.array(right_fit)
        # Calculate average fit to line
        right_line.avg_fit = np.sum(np.asarray(right_line.recent_fit), axis=0) / len(right_line.recent_fit)
        right_line.radius = right_rad

    if left_line.detected and right_line.detected:
        centre_offset = find_offset()
        left_line.centre_offset = centre_offset
        right_line.centre_offset = centre_offset

    if visualise:
        # Draw lines with average fit coefficients
        plot_lanes(img_out, left_line.current_fit, right_line.current_fit)

    return left_line.current_fit, right_line.current_fit

def check_lane_fit(lefty, leftx, righty, rightx):
    # Assume lane lines have been detected and check if true
    left_line.detected = True
    right_line.detected = True

    # Set empty lists for line fit coefficients
    left_fit  = []
    right_fit = []
    left_rad  = min_lane_radius
    right_rad = min_lane_radius
    # If minimum pixels of left line not detected in current image
    if leftx.size < min_pixels_line:
        left_line.detected = False
    else:
        # Fit a second order polynomial to detected left line pixels
        left_fit = np.polyfit(lefty, leftx, left_line.fit_degree)
        # Check direction of curve
        if left_fit[0]*left_line.current_fit[0] < 0:
            if left_fit[0]*left_line.avg_fit[0] < 0:
                left_fit = fit_gain*left_line.current_fit + (1-fit_gain)*left_fit
        # Calculate left line radius
        left_rad, _ = find_curvature(lefty, leftx, 'left')
        # Check if radius is within deviation limits from previously calculated radius
        if (left_rad/left_line.radius > max_radius_ratio) or (left_line.radius/left_rad > max_radius_ratio):
            left_line.detected = False
        # Check if radius is greater than minimum standard U.S. highway radius
        if left_rad < min_lane_radius:
            left_line.detected = False

    # If minimum pixels of right line not detected in current image
    if rightx.size < min_pixels_line:
        right_line.detected = False
        # Increment missed frames counter
        right_line.n_missed_frames += 1
    else:
        # Fit a second order polynomial to detected right line pixels
        right_fit = np.polyfit(righty, rightx, right_line.fit_degree)
        # Check direction of curve
        if right_fit[0] * right_line.current_fit[0] < 0:
            if right_fit[0] * right_line.avg_fit[0] < 0:
                right_fit = fit_gain*right_line.current_fit + (1-fit_gain)*right_fit
        # Calculate right line radius
        _, right_rad = find_curvature(righty, rightx, 'right')
        # Check if radius is within deviation limits from previously calculated radius
        if(right_rad/right_line.radius > max_radius_ratio) or (right_line.radius/right_rad > max_radius_ratio):
            right_line.detected = False
        # Check if radius is greater than minimum standard U.S. highway radius
        if right_rad < min_lane_radius:
            right_line.detected = False

    # Check if lane width is within standard minimum width
    if left_line.detected and right_line.detected:
        # Calculate x-cordinate of start of left lane
        left_line_xm = left_fit[0] * img_height_crop ** 2 + left_fit[1] * img_height_crop + left_fit[2]
        left_line_xm *= xm_per_pix
        # Calculate x-cordinate of start of right lane
        right_line_xm = right_fit[0] * img_height_crop ** 2 + right_fit[1] * img_height_crop + right_fit[2]
        right_line_xm *= xm_per_pix
        current_lane_width = right_line_xm - left_line_xm
        if (current_lane_width) < min_lane_width:
            left_line.detected  = False
            right_line.detected = False

    return left_fit, right_fit, left_rad, right_rad

def find_curvature(liney, linex, lane_line):
    # Y-coordinate of point to calculate the curvature at
    y_eval = int(img_height/2)
    # Fit new polynomials to x,y in world space
    left_rad  = -1.0
    right_rad = -1.0
    if lane_line == 'left':
        left_fit_wc  = np.polyfit(liney * ym_per_pix, linex * xm_per_pix, left_line.fit_degree)
        # Calculate the new left radius of curvature
        left_rad = ((1 + (2 * left_fit_wc[0] * y_eval * ym_per_pix + left_fit_wc[1]) ** 2) ** 1.5) \
                     /np.absolute(2 * left_fit_wc[0])
    if lane_line == 'right':
        right_fit_wc = np.polyfit(liney * ym_per_pix, linex * xm_per_pix, right_line.fit_degree)
        # Calculate the new right radius of curvature
        right_rad = ((1 + (2 * right_fit_wc[0] * y_eval * ym_per_pix + right_fit_wc[1]) ** 2) ** 1.5) \
                     /np.absolute(2 * right_fit_wc[0])
    return left_rad, right_rad

def find_offset():
    y_base = img_height - 10
    # Calculate x coordinate of base of left and right lines
    leftx_base = left_line.current_fit[0] * (y_base) ** 2 + left_line.current_fit[1] * y_base \
                 + left_line.current_fit[2]
    rightx_base = right_line.current_fit[0] * (y_base) ** 2 + right_line.current_fit[1] * y_base + \
                  right_line.current_fit[2]
    # Calculate x coordinate of centre of lane
    midx_base = (rightx_base - leftx_base) // 2 + leftx_base
    centre_offset = (midx_base - img_width//2) * xm_per_pix
    return centre_offset

def plot_lanes(img_out, left_fit, right_fit):
    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(img_out)
    # Generate x and y values for plotting
    y_fit = np.linspace(0, img_out.shape[0]-1, img_out.shape[0])
    if len(left_fit) == (left_line.fit_degree+1):
        leftx_fit = left_fit[0] * y_fit ** 2 + left_fit[1] * y_fit + left_fit[2]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([leftx_fit - win_margin, y_fit]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([leftx_fit + win_margin, y_fit])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        # Draw left line pixels on output image
        img_out[left_line.ally, left_line.allx] = [255, 0, 0]
        # Draw lane lines based on current fit
        plt.plot(leftx_fit, y_fit, color='yellow')

    if len(right_fit) == (right_line.fit_degree+1):
        rightx_fit = right_fit[0] * y_fit ** 2 + right_fit[1] * y_fit + right_fit[2]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        right_line_window1 = np.array([np.transpose(np.vstack([rightx_fit - win_margin, y_fit]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rightx_fit + win_margin, y_fit])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        # Draw right line pixels on output image
        img_out[right_line.ally, right_line.allx] = [0, 0, 255]
        # Draw lane lines based on current fit
        plt.plot(rightx_fit, y_fit, color='yellow')

    result = cv2.addWeighted(img_out, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.xlim(0, img_out.shape[1])
    plt.ylim(img_out.shape[0], 0)
    plt.show()
