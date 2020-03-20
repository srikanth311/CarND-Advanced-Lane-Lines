import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

from _PlotMyImages import PlotMyImages as plot_my_images

class MyImageProcessorUtil(object):

    @staticmethod
    def cal_undistort(all_variables):
        all_variables.undistorted_image = cv2.undistort(all_variables.image, all_variables.camera_matrix, all_variables.dist_coefficient, None, all_variables.camera_matrix)
        # print(all_variables.undistorted_image.shape)
        #plt.imshow(all_variables.undistorted_image)
        #plt.show()
        # Showing original and undisorted images for a quick test. Uncomment the below line to see the output.
        #plot_my_images.plot_my_images(all_variables.image, all_variables.undistorted_image)
        #cv2.imwrite("output_images/undistorted_images/" + os.path.basename(all_variables.input_image_path) + ".png", all_variables.undistorted_image)
    
    @staticmethod
    def generate_gray_and_hls_images_from_undistorted_images(all_variables):
        all_variables.gray_image = cv2.cvtColor(all_variables.undistorted_image, cv2.COLOR_RGB2GRAY)
        all_variables.hls_image = cv2.cvtColor(all_variables.undistorted_image, cv2.COLOR_RGB2HLS)
        
        # Show Gray image and HLS image.
        #plot_my_images.plot_my_images(all_variables.gray_image, all_variables.hls_image, "Gray Image", "HLS_Image")

    @staticmethod
    def processed_schannel_image_from_hls_image(all_variables):
        # Threshold color channel
        # uainf the same thresholds from the example code in the lesson.
        #s_thresh_min = 170
        #s_thresh_max = 255

        s_thresh_min = 90
        s_thresh_max = 255

        #Since my gray image is stored in a variable in _AllVariablesPipeline class, reading it from there.
        s_channel = all_variables.hls_image[:,:,2]
        all_variables.schannel_binary_image = np.zeros_like(s_channel)
        all_variables.schannel_binary_image[(s_channel > s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        #plot_my_images.plot_an_image(all_variables.schannel_binary_image)
    
    @staticmethod
    def processed_rchannel_image_from_hls_image(all_variables):
        # Threshold color channel
        # uainf the same thresholds from the example code in the lesson.
        #s_thresh_min = 170
        #s_thresh_max = 255

        r_thresh_min = 200
        r_thresh_max = 255

        #Since my gray image is stored in a variable in _AllVariablesPipeline class, reading it from there.
        r_channel = all_variables.hls_image[:,:,2]
        all_variables.rchannel_binary_image = np.zeros_like(r_channel)
        all_variables.rchannel_binary_image[(r_channel > r_thresh_min) & (r_channel <= r_thresh_max)] = 1
        #plot_my_images.plot_an_image(all_variables.schannel_binary_image)

    @staticmethod
    def process_sobel_absolute_from_gray_image(all_variables):
        thresh_min = 20
        thresh_max = 100

        #thresh_min = 50
        #thresh_max = 255

        #thresh_min = 10
        #thresh_max = 160

        # Calcluate the derivative in the x direction        
        sobelx = cv2.Sobel(all_variables.gray_image, cv2.CV_64F, 1, 0)
        # Calculate the derivative in y direction.
        sobely = cv2.Sobel(all_variables.gray_image, cv2.CV_64F, 0, 1)
        # Calculate the absolute value of the x derivative
        abs_sobelx = np.absolute(sobelx)
        # Convert the absolute value image to 8-bit
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        # Create a binary threshold to select pixels based on gradient strength.
        sxbinary = np.zeros_like(scaled_sobel)
        # pixels will have a value of 1 or 0 based on the strength of the x gradient.
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        all_variables.sobel_binary_image = sxbinary
        # For testing - plot the image.
        #plot_my_images.plot_an_image(all_variables.sobel_binary_image)

    @staticmethod
    def process_soble_direction_of_the_gradient(all_variables):
        #sobel_kernel = 15
        sobel_kernel = 3
        #direction_threshold = (0.7, 1.3)
        direction_threshold = (0, np.pi/2)
        #direction_threshold = (0.79, 1.20)
        # Take gradient in x and y separately
        # Calcluate the derivative in the x direction        
        sobelx = cv2.Sobel(all_variables.gray_image, cv2.CV_64F, 1, 0)
        # Calculate the derivative in y direction.
        sobely = cv2.Sobel(all_variables.gray_image, cv2.CV_64F, 0, 1)
        # Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        arctan = np.arctan2(abs_sobely, abs_sobelx)
        # Create a binary mask where direction thresholds are met
        sobel_direction_binary_output = np.zeros_like(arctan)
        # Return this mask as your binary_output image
        sobel_direction_binary_output[(arctan >= direction_threshold[0]) & (arctan <= direction_threshold[1])] = 1
        all_variables.sobel_direction_binary_image = sobel_direction_binary_output
        # For testing - plot the image.
        #plot_my_images.plot_an_image(all_variables.sobel_direction_binary_image)

    @staticmethod
    def process_sobel_magnitude_of_the_gradient(all_variables):
        sobel_mag_kernel = 3
        #sobel_mag_threshold = (50, 255)
        sobel_mag_threshold = (30, 100)
        # Take gradient in x and y separately
        # Calcluate the derivative in the x direction        
        sobelx = cv2.Sobel(all_variables.gray_image, cv2.CV_64F, 1, 0)
        # Calculate the derivative in y direction.
        sobely = cv2.Sobel(all_variables.gray_image, cv2.CV_64F, 0, 1)
        # 3) Calculate the magnitude 
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
        # Create a copy and apply the threshold
        sobel_mag_binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        sobel_mag_binary_output[(scaled_sobel >= sobel_mag_threshold[0]) & (scaled_sobel <= sobel_mag_threshold[1])] = 1
        all_variables.sobel_magnitude_binary_image = sobel_mag_binary_output
        # For testing - plot the image.
        #plot_my_images.plot_an_image(all_variables.sobel_magnitude_binary_image)

    @staticmethod
    def combining_all_thresholds(all_variables):
        combined_binary_image_local = np.zeros_like(all_variables.sobel_binary_image)
        combined_binary_image_local[      ((all_variables.sobel_binary_image == 1) 
                                        | ((all_variables.sobel_direction_binary_image == 1) & (all_variables.sobel_magnitude_binary_image == 1))) 
                                        | (all_variables.schannel_binary_image == 1) 
                                        | (all_variables.rchannel_binary_image == 1)
                                   ] = 1
        all_variables.combined_binary_image = combined_binary_image_local
        # For testing - plot the image.
        #plot_my_images.plot_an_image(all_variables.combined_binary_image)
    
    @staticmethod
    def perspective_transofrm_of_the_combined_binary_image(all_variables):
        image_size = (all_variables.combined_binary_image.shape[1], all_variables.combined_binary_image.shape[0])
        src = np.float32([[200, 720],[1100, 720],[595, 450],[685, 450]])
        dst = np.float32([[300, 720],[980 , 720],[300, 0  ],[980, 0  ]])

        all_variables.perspective_transform_matrix = cv2.getPerspectiveTransform(src, dst)
        all_variables.perspective_inverse_matrix = cv2.getPerspectiveTransform(dst, src)
        all_variables.warped_binary_image = cv2.warpPerspective(all_variables.combined_binary_image, all_variables.perspective_transform_matrix,
                                                                image_size, flags=cv2.INTER_LINEAR)
        # For testing - plot the image.
        #plot_my_images.plot_an_image(all_variables.warped_binary_image)
        #cv2.imwrite("output_images/perspective_transofrm_of_the_combined_binary_image"  + ".jpg", all_variables.warped_binary_image)

    
