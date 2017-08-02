import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# This code is taken from the Udacity camera calibration notebook
# https://github.com/udacity/CarND-Camera-Calibration

def calibrate_camera(img_path="camera_cal/calibration*.jpg", params_path = "camera_cal/camera_params.p"):
    nx = 9  # enter the number of inside corners in  x
    ny = 6  # enter the number of inside corners in y
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(img_path)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(img_dir+write_name, img)

    #cv2.destroyAllWindows()

    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print('Calibration complete')

    # Save the camera calibration result for later use
    camera_params = {}
    camera_params["intrinsic"] = mtx
    camera_params["distortion"] = dist
    pickle.dump( camera_params, open( params_path, "wb" ) )
    print("Camera parameters saved")

def test_camera(img_path="camera_cal/calibration2.jpg", params_file='./camera_cal/camera_params.p'):
    with open(params_file, mode='rb') as f:
        params = pickle.load(f)
    mtx = params['intrinsic']
    dist = params['distortion']

    # Test undistortion on an image
    img = cv2.imread(img_path)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt._show()