import pickle
import glob
from functools import reduce
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from process_image import *
from detect_lanes import *

# File and directory paths
params_file   = 'camera_params.p'
video_input    = 'project_video.mp4'
video_output  = 'project_video_output.mp4'
img_dir       = 'test_images/'
img_file      = 'straight_lines1.jpg'
video_img_dir =  img_dir+'project_video/'
# Pipeline to process camera image to isolate lane markings
def pipeline(img):
    visualise = True
    img_undist = undistort_image(img, mtx=mtx, dist=dist, visualise=visualise)
    img_thresh = threshold_image(img_undist, visualise=visualise)
    img_top = view_road_top(img_thresh, img, visualise=True)
    find_lane_pixels(img_top, visualise=True)
    return img_top

# Read camera intrinsic matrix and distortion coefficients
# Created initially with calibrate_camera.py
with open(params_file, mode='rb') as f:
    params = pickle.load(f)
mtx  = params['intrinsic']
dist = params['distortion']

# Number of past images to store for lane detection
n_steps = 10
img_height = 720
img_width = 1280
img_channels = 3
img_buffer = np.zeros((n_steps, img_height, img_width, img_channels), dtype=np.uint8)

# Video is at 25 FPS
clip = VideoFileClip(video_input).subclip(0,2)
video_times = np.linspace(0, 1, n_steps)
for vt in video_times:
        video_img_file = video_img_dir + 'video{}.jpg'.format(vt)
        clip.save_frame(video_img_file, vt)

#clip_output = clip.fl_image(pipeline) #NOTE: this function expects color images!!
#clip_output.write_videofile(video_output, audio=False)

# Make a list of calibration images
images = glob.glob(video_img_dir+'video*.jpg')
i=0
for image in images:
    img = mpimg.imread(image)
    img_buffer[i] = img
    i+=1
img = reduce(np.maximum, img_buffer)
plt.imshow(img)
plt.show()
pipeline(img)

