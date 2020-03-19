import numpy as np
import cv2
import matplotlib.pyplot as plt

from _AllVariablesInPipeline import AllVariablesInPipeline
from _GlobalLaneVariables import GlobalLaneVariables as GLV
from _PlotMyImages import PlotMyImages as plot_my_images

class FindingLanes(object):

    @staticmethod
    def finding_lane_pixels(all_variables):
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero_points = all_variables.warped_binary_image.nonzero()

        if FindingLanes.fit_lane_pixels(nonzero_points):
            return

        image_height = all_variables.warped_binary_image.shape[0]
        # Take a histogram of the bottom half of the image
        histogram = np.sum(all_variables.warped_binary_image[all_variables.warped_binary_image.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((all_variables.warped_binary_image, all_variables.warped_binary_image, all_variables.warped_binary_image))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[100:midpoint]) + 100
        rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

        
        all_variables.left_lane_window_positions, left_lane_points_x , left_lane_points_y, left_out_img = FindingLanes.step_through_sliding_windows(nonzero_points, leftx_base, image_height, out_img)
        all_variables.right_lane_window_positions, right_lane_points_x , right_lane_points_y, right_out_img = FindingLanes.step_through_sliding_windows(nonzero_points, rightx_base, image_height, left_out_img)

        GLV.global_left_lane_points_x.append(left_lane_points_x)
        GLV.global_left_lane_points_y.append(left_lane_points_y)
        GLV.global_right_lane_points_x.append(right_lane_points_x)
        GLV.global_right_lane_points_y.append(right_lane_points_y)
        print("In find lane points")
        

        # For testing - plot the image.
        # right_out_image contains both left and right as we are passing the left_out_img 2nd time to "step_through_sliding_windows"
        # plot_my_images.plot_an_image(right_out_img)
  
    @staticmethod
    def fit_lane_pixels(nonzero_points):
        if GLV.global_left_lane_poly is None:
            return False

        margin = 100
        minpoints = 10
        Y=np.array(nonzero_points[0])
        X=np.array(nonzero_points[1])

        poly = GLV.global_left_lane_poly
        left_fit_x = poly[0]*(Y**2) + poly[1]*Y + poly[2]
        poly = GLV.global_right_lane_poly
        right_fit_x = poly[0]*(Y**2) + poly[1]*Y + poly[2]

        left_indices = (X > (left_fit_x - margin)) & (X < (left_fit_x + margin))
        right_indices = (X > (right_fit_x - margin)) & (X < (right_fit_x + margin))

        if( len(left_indices) < minpoints or len(right_indices) < minpoints):
            return False

        if(len(X[left_indices]) > 5):
            GLV.global_left_lane_points_x.append(X[left_indices])
            GLV.global_left_lane_points_y.append(Y[left_indices])
            GLV.global_left_lane_poly = np.polyfit(X[left_indices],Y[left_indices],2)

        if(len(X[right_indices]) > 5):
            GLV.global_right_lane_points_x.append(X[right_indices])
            GLV.global_right_lane_points_y.append(Y[right_indices])
            GLV.global_right_lane_poly = np.polyfit(X[right_indices],Y[right_indices],2)

        return True

    @staticmethod
    def step_through_sliding_windows(nonzero_points, base_x_value, input_image_height, out_img):
        # print("image height in sliding windows is: " + str(input_image_height))
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(input_image_height//nwindows)
        nonzeroy = np.array(nonzero_points[0])
        nonzerox = np.array(nonzero_points[1])
        points_x=[]
        points_y=[]
        window_positions=[]

        for window in range(nwindows):
            x1 = base_x_value - margin
            x2 = base_x_value + margin
            y1 = input_image_height - (window+1)*window_height
            y2 = input_image_height - window*window_height

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(x1, y1),(x2, y2),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_line_indices = ((nonzeroy >= y1) & (nonzeroy < y2) & (nonzerox >= x1) & (nonzerox < x2)).nonzero()[0]
            points_x.append(nonzerox[good_line_indices])
            points_y.append(nonzeroy[good_line_indices])
            window_positions.append([x1, y1, x2, y2])
            
            # If you found > minpix pixels, recenter next window on their mean position,
            # otherwise it will use the same previous base_x_value for the next window as well.
            if len(good_line_indices) > minpix:
                base_x_value = np.int(np.mean(nonzerox[good_line_indices]))
        
        points_x = np.concatenate(points_x)
        points_y = np.concatenate(points_y)

        return window_positions, points_x , points_y, out_img

    @staticmethod
    def generate_polynomial_plot_points(all_variables):
        image_height = all_variables.undistorted_image.shape[0]
        all_variables.left_lane_points_x = np.concatenate(GLV.global_left_lane_points_x)
        all_variables.left_lane_points_y = np.concatenate(GLV.global_left_lane_points_y)
        all_variables.right_lane_points_x = np.concatenate(GLV.global_right_lane_points_x)
        all_variables.right_lane_points_y = np.concatenate(GLV.global_right_lane_points_y)

        # In a given image (in our case our image height is 720 pixels), linspace gives 0 to 719 values like 0, 1, 2, --- upto 719.
        ploty = np.linspace(0, image_height-1, image_height)

        # So for every point in y axis which we get from linspace command,
        # we need to find corresponding x value by using polyfit function.
        # polyfit function gives "y" values in a given polynomial equation
        # for eg: in a second degree polynomial equation "x = Ay**2+By+C",
        # it gives A, B and c values.
        # Once we get these values, we can can draw the poly line.

        # Apply 2nd degree polynomial
        # Apply the same for two plots.
        # The result contains 3 values from the polyfit function to use it in "y=Ax**2+Bx+c"
        all_variables.left_lane_poly = np.polyfit(all_variables.left_lane_points_y, all_variables.left_lane_points_x, 2)
        print(all_variables.left_lane_poly)
        all_variables.right_lane_poly = np.polyfit(all_variables.right_lane_points_y, all_variables.right_lane_points_x, 2)
        print(all_variables.right_lane_poly)

        all_variables.left_lane_plot_x = all_variables.left_lane_poly[0]*ploty**2 + all_variables.left_lane_poly[1]*ploty + all_variables.left_lane_poly[2]
        all_variables.left_lane_plot_y = ploty

        all_variables.right_lane_plot_x = all_variables.right_lane_poly[0]*ploty**2 + all_variables.right_lane_poly[1]*ploty + all_variables.right_lane_poly[2]
        all_variables.right_lane_plot_y = ploty

        GLV.global_left_lane_poly = all_variables.left_lane_poly
        GLV.global_right_lane_poly = all_variables.right_lane_poly
        
        #plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)

    @staticmethod
    def measure_curvature(all_variables):
        image_height = all_variables.undistorted_image.shape[0]
        y_eval = image_height-1  # 720p video/image, so last (lowest on screen) y index is 719
        y_meters_per_pix = 30/image_height # meters per pixel in y dimension
        x_meters_per_pix = 3.7/700 # meters per pixel in x dimension

        left_poly  = np.polyfit(all_variables.left_lane_points_y*y_meters_per_pix, all_variables.left_lane_points_x*x_meters_per_pix, 2)
        right_poly = np.polyfit(all_variables.right_lane_points_y*y_meters_per_pix, all_variables.right_lane_points_x*x_meters_per_pix, 2)

        # Calculation of R_curve (radius of curvature)
        left_curve_radius = ((1 + (2*left_poly[0]*y_eval*y_meters_per_pix + left_poly[1])**2)**1.5) / np.absolute(2*left_poly[0])
        right_curve_radius = ((1 + (2*right_poly[0]*y_eval*y_meters_per_pix + right_poly[1])**2)**1.5) / np.absolute(2*right_poly[0])

        all_variables.radius_of_curvature = (left_curve_radius + right_curve_radius)/2

    @staticmethod
    def measure_vehicle_offset(all_variables):

        # Calculate vehicle center offset in pixels
        bottom_y = 719
        bottom_x_left = all_variables.left_lane_poly[0]*(bottom_y**2) + all_variables.left_lane_poly[1]*bottom_y + all_variables.left_lane_poly[2]
        bottom_x_right = all_variables.right_lane_poly[0]*(bottom_y**2) + all_variables.right_lane_poly[1]*bottom_y + all_variables.right_lane_poly[2]

        all_variables.vehicle_offset = all_variables.undistorted_image.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

        # Convert pixel offset to meters
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        all_variables.vehicle_offset *= xm_per_pix