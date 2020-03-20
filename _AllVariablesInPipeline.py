import collections

class AllVariablesInPipeline(object):
    def __init__(self, input_image, input_image_path, camera_matrix, dist_coefficient):
        self.image = input_image
        self.input_image_path = input_image_path
        self.camera_matrix = camera_matrix
        self.dist_coefficient = dist_coefficient
        self.gray_image = None
        self.hls_image = None

        
        self.undistorted_image = None
        self.sobel_binary_image = None
        self.schannel_binary_image = None
        self.rchannel_binary_image = None
        self.sobel_magnitude_binary_image = None
        self.sobel_direction_binary_image = None
        self.combined_binary_image = None
        self.warped_binary_image = None

        self.perspective_transform_matrix = None
        self.perspective_inverse_matrix = None

        self.img_sliding_window = None
        self.color_warp = None
        self.color_unwarped = None
        self.img_processed = None

        #Processed by Finding Lanes class
        self.left_lane_points_x = None
        self.left_lane_points_y = None
        self.left_lane_window_positions = None

        self.right_lane_points_x = None
        self.right_lane_points_y = None
        self.right_lane_window_positions = None

        self.left_lane_poly = None
        self.left_lane_plot_x = None
        self.left_lane_plot_y = None
        
        self.right_lane_poly = None
        self.right_lane_plot_x = None
        self.right_lane_plot_y = None

        self.radius_of_curvature = None
        self.vehicle_offset = None