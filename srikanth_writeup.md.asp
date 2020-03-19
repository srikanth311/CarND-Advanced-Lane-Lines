## Advanced Lane Finding.

#### Overview
##### In this document we will focus on understanding various lane finding mechanisms/procedures from a computer vision perspective. These mechanisms will be very useful especially in finding the curved lanes and the lanes with different color other than white color. The objective of this project is to identify the lines in a video feed along with the radius of the curvature and the offset from the center position of the lane.

##### Note: I have used the example codes from the class and also researched online for better understanding while working on this project.

#### The main entry to this project is either : ```_WrapperPipelineImage.py or _WrapperPipelineImage.ipynb``` to process a single image.
#### To process the videp, the main entry program is : ```_WrapperPipelineVideo.ipynb or _WrapperPipelineVideo.py```.
#### The video file is generated and stored with the name " ```generated_output_video.mp4``` in the root folder.

#### Overall Steps
##### 1. Calculate the camera calibration and correcting the distortion.
##### 2. Apply Color and Gradient thresholding
##### 3. Apply Perspective transformation
##### 4. Find the points of the lane
##### 5. Measure the curvature of the radius and vehicle offset from the lane center

#### Calculate the camera calibration and correcting the distortion.
##### The ultimate goal in this first section is to measure some of the quantities that need to be known in order to control the car. For eg: to steer a car, we need to measure how much the lane is curving. For this we need to map out the lens in the camera images, after transforming them to a different perspective. To get to this perspective transformation, we first need to correct for the effect of image distrotion. Some of the objects in the images, especially ones near the edges, can get stretched or skewed in various ways and we need to correct them.
##### A camera looks at 3D objects in the real world and transforms them into a 2D image. This transformation is not perfect. If the lane is distroted, we will get the wrong measurement while steering it.
##### In this exercise, we will be using the images from the camera_cal to determine the camera matrix and the distortion coefficients and will be used these outputs to undistort images.
##### Code snippet to convert Grayscale images

```python
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

```

##### Distroted and Undistorted images
<table>
  <tr>
    <td><img src="camera_cal/calibration1.jpg" width=370 height=170></td>
    <td><img src="output_images/undistorted_images/calibration1.jpg.png" width=370 height=170></td>
  </tr>
 </table>

 #### Apply Color and Gradient thresholding
 ##### The single color map can clearly identify the white lines, but the yellow lines will not be detected. For this we apply various color and gradient thresholding techniques to detect the lanes and we can create the edges. Finally we will create a binary image at the end of this section.
 ##### After applying various thresholds, i have found S-Channel of HSL color map can clearly identify the yellow channels. Also, I have used sobel operators for the gradient thresholding and this can detect the other edges in the image.

##### Code snippet
```python
    @staticmethod
    def cal_undistort(all_variables):
        all_variables.undistorted_image = cv2.undistort(all_variables.image, all_variables.camera_matrix, all_variables.dist_coefficient, None, all_variables.camera_matrix)
        cv2.imwrite("output_images/undistorted_images/" + os.path.basename(all_variables.input_image_path) + ".png", all_variables.undistorted_image)
```
```python   
    @staticmethod
    def generate_gray_and_hls_images_from_undistorted_images(all_variables):
        all_variables.gray_image = cv2.cvtColor(all_variables.undistorted_image, cv2.COLOR_RGB2GRAY)
        all_variables.hls_image = cv2.cvtColor(all_variables.undistorted_image, cv2.COLOR_RGB2HLS)
```       
```python
    @staticmethod
    def processed_schannel_image_from_hls_image(all_variables):
        # Threshold color channel
        # uainf the same thresholds from the example code in the lesson.
        s_thresh_min = 170
        s_thresh_max = 255

        #Since my gray image is stored in a variable in _AllVariablesPipeline class, reading it from there.
        s_channel = all_variables.hls_image[:,:,2]
        all_variables.schannel_binary_image = np.zeros_like(s_channel)
        all_variables.schannel_binary_image[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        #plot_my_images.plot_an_image(all_variables.schannel_binary_image)
```
```python
    @staticmethod
    def process_sobel_absolute_from_gray_image(all_variables):
        thresh_min = 20
        thresh_max = 100

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
        # plot_my_images.plot_an_image(all_variables.sobel_binary_image)
```
```python
    @staticmethod
    def process_soble_direction_of_the_gradient(all_variables):
        sobel_kernel = 15
        direction_threshold = (0.7, 1.3)
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
```
```python
    @staticmethod
    def process_sobel_magnitude_of_the_gradient(all_variables):
        sobel_mag_kernel = 3
        sobel_mag_threshold = (50, 255)
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
        # plot_my_images.plot_an_image(all_variables.sobel_magnitude_binary_image)
```
```python
    @staticmethod
    def combining_all_thresholds(all_variables):
        combined_binary_image_local = np.zeros_like(all_variables.sobel_binary_image)
        combined_binary_image_local[ ((all_variables.sobel_binary_image == 1) | ((all_variables.sobel_direction_binary_image == 1) & (all_variables.sobel_magnitude_binary_image == 1))) | (all_variables.schannel_binary_image == 1) ] = 1
        all_variables.combined_binary_image = combined_binary_image_local
        # For testing - plot the image.
        # plot_my_images.plot_an_image(all_variables.combined_binary_image)
```
##### Output after doing the color and gradient thresholds
<table>
  <tr>
    <td><img src="output_images/project2_color_gradient_test2_output.png"></td>
  </tr>
 </table>

 #### Apply Perspective transformation
 ##### In an image, perspective is a phenomenon where an object appears smaller the farther away it is from a viewpoint like a camera. For eg, the lane on a road looks smaller and smaller the farther away it gets from the camera. A perspective transform warps the image and effectively drags points towards or pushes them away from the camera to change the apparent perspective. This is typically changing the lane view from camera point of view to a bird's eye view. Finding a curvature of a lane is easier to perform on a bird's eye view of an image.

 ##### Code snippet
```python
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
```

#### Find the points of the lane
##### From the warp perspective of an image, we can detect the lane points.

#### Code Snippet for iddentifying the region of interest
```python
def finding_lane_pixels(all_variables):
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero_points = all_variables.warped_binary_image.nonzero()
        image_height = all_variables.warped_binary_image.shape[0]
        # Take a histogram of the bottom half of the image
        histogram = np.sum(all_variables.warped_binary_image[all_variables.warped_binary_image.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((all_variables.warped_binary_image, all_variables.warped_binary_image, all_variables.warped_binary_image))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
       
        all_variables.left_lane_window_positions, left_lane_points_x , left_lane_points_y, left_out_img = FindingLanes.step_through_sliding_windows(nonzero_points, leftx_base, image_height, out_img)
        all_variables.right_lane_window_positions, right_lane_points_x , right_lane_points_y, right_out_img = FindingLanes.step_through_sliding_windows(nonzero_points, rightx_base, image_height, left_out_img)
```
```python
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
```
##### Warped and Sliding window with polynomial curve images.
<table>
  <tr>
    <td><img src="output_images/project2_warped_sliding_test2_output.png" ></td>
  </tr>
 </table>

 #### Measure the curvature of the radius and vehicle offset from the lane center
 ##### For self driving cars, finding the radius of the curvature is important to determine the steering angle of the vehicle.
 ```python
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

    def measure_vehicle_offset(all_variables):

        # Calculate vehicle center offset in pixels
        bottom_y = 719
        bottom_x_left = all_variables.left_lane_poly[0]*(bottom_y**2) + all_variables.left_lane_poly[1]*bottom_y + all_variables.left_lane_poly[2]
        bottom_x_right = all_variables.right_lane_poly[0]*(bottom_y**2) + all_variables.right_lane_poly[1]*bottom_y + all_variables.right_lane_poly[2]

        all_variables.vehicle_offset = all_variables.undistorted_image.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

        # Convert pixel offset to meters
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        all_variables.vehicle_offset *= xm_per_pix
 ``` 
 
#### Here is the complete set of images generated from the pipeline for processing a given single image.
<table>
  <tr>
    <td><img src="output_images/project2_pipeline_test2_output.png" ></td>
  </tr>
 </table>

#### Testing on video 1
##### After applying the above pipeline, the video looks like this.
<video width="320" height="240" controls>
  <source src="generated_output_video.mp4" type="generated_output_video.mp4">
</video>

### Shortcomings
  * The current pipeline may have problems when it is rainy day or snowy day as the lines might not be that much clearer.
  * I have used hard coded values to find the lane points, If the camera view changes, i might have to change these values.
  * If there are no lane markings on the road, this will not work.
### Future improvements
  * I have used trail and error method for the thresholds values, We can improve this to find the better optimal thresholds by creating a tool.
   