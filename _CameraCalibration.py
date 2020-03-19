import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import cv2
import glob
import os

class CameraCalibration(object):

    check_camera_calibrated = None

    def __init__(self):
        self.camera_matrix = None
        self.dist_coefficient = None

    def calibrate_camera(self):
        # number of chess board corners to detect from the camera_cal images is 9x6.
        nx = 9
        ny = 6

        # map the coordinates of the corners in this 2D image which I called image_points[], to the 3D coordinates of the 
        # real, undistorted chess board corners which are called object_points[].
        # The object points will all be the same, just the known object coordinates of the chess board corners for an 9X6 board in our case.
        # These points will be the 3D coordinates, x,y and z from the top left corner (0,0,0) tp bottom right (8,5,0)
        # The z coordinate will be "ZERO" for every point, since the board is on a flat image plane.
        object_points = [] # 3D points in real world space
        image_points = [] # 2D points in the image plane.

        objpoints = np.zeros((nx*ny, 3), np.float32)
        # Z columns is zero, but for first two columns x and y, will use mgrid function of numpy to generate the coordinates that we need.
        # mgrid returns the coordinate values for a given grid size and shape these coordinates back to two columns, one for x and one for y.
        objpoints[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        
        # To create image points look at the distorted calibration image and detect the corners of the board.
        # For this opencv provides a function to detect the chess board corners. (findChessboardCorners)
        # It returns the corners found in a gray scale image.

        # We can read all the images by importing the glob api.
        # glob() helps to read the images with a consistent file name.

        images = glob.glob("./camera_cal/*")
        for idx, filename in enumerate(images):
            img = mpimg.imread(filename)
            gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # findChessboardCorners takes the gray scale image, along with the dimension of the chess board corners, in this case nx, and ny
            ret_pattern_found, corner_points = cv2.findChessboardCorners(gray_image, (nx, ny), None)

            # if the findChessboardCorners function detects the corners, then append the corner points to image_pointsp[].
            # Also add the prepared objects points (object_points) to the points array.
            # These object points will be the same for all the calibrated images.
            if ret_pattern_found == True:
                object_points.append(objpoints)
                image_points.append(corner_points)
                # Draw the detected corners using "drawChessboardCorners()"
                # This function takes our image, corner dimension and corner points.
                cv2.drawChessboardCorners(img, (nx, ny), corner_points, ret_pattern_found)
                cv2.imwrite("output_images/chessboard_images/chessboardcorners_" + os.path.basename(filename) + ".png", img)
            
        #Get the image size (any image)
        image_size = mpimg.imread(images[0]).shape[0:2]
        # Use object points and image points with opencv's calibrateCamera function to calibrate this camera (using all the images that we have).
        # It takes object points, image points, and the shape of the image.
        # Using these inputs it calculates and returns the "distortion coefficients" and the "camera matrix" that we need to transform 3D objects
        # to 2D image points.
        # It also returns the position of the camera in the world with values of rotation and translation vectors.
        ret, self.camera_matrix, self.dist_coefficient, rotation_vector, translation_vector = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    @staticmethod
    def get_calibrated_camera():
        camera_calibration = None
        if CameraCalibration.check_camera_calibrated is not None:
            print("in is not none check")
            camera_calibration = CameraCalibration.check_camera_calibrated
        #print("In camera calibration")
        # Check if we already generated the pickle file
        # If we run the code on several input images, we can just calculate the camera_matrix and dist_coefficients only once.
        elif os.path.isfile("camera_calibrated_pickle_file.p"):
            with open('camera_calibrated_pickle_file.p', 'rb') as input_pickle_file:
                print("loading existing pickle file")
                camera_calibration = pickle.load(input_pickle_file)
        else:
            camera_calibration = CameraCalibration()
            camera_calibration.calibrate_camera()

            #Use pickle to save the camera_calibration object and dump the pickle file with camera matrix, dist_coefficients
            with open("camera_calibrated_pickle_file.p", "wb") as pickle_file:
                print("writing calibration file")
                pickle.dump(camera_calibration, pickle_file)
        
        return camera_calibration


