from _AllVariablesInPipeline import AllVariablesInPipeline
from _MyImageProcessorUtil import MyImageProcessorUtil as my_image_processor_util
from _FindingLanes import FindingLanes as finding_lanes
from _PlotMyImages import PlotMyImages as plot_my_images
import time

def my_pipeline(image, image_path, camera_calibration, singleImageExecution=False):
    all_variables_in_pipeline = AllVariablesInPipeline(image, image_path, camera_calibration.camera_matrix, camera_calibration.dist_coefficient)

    # Step1 - Undistortion of the input image.
    my_image_processor_util.cal_undistort(all_variables_in_pipeline)
    

    # Step2 - 
    my_image_processor_util.generate_gray_and_hls_images_from_undistorted_images(all_variables_in_pipeline)

    # Step 3
    my_image_processor_util.processed_schannel_image_from_hls_image(all_variables_in_pipeline)
    
    # Step 4
    my_image_processor_util.process_sobel_absolute_from_gray_image(all_variables_in_pipeline)

    # Step 5
    my_image_processor_util.process_soble_direction_of_the_gradient(all_variables_in_pipeline)

    # Step 6
    my_image_processor_util.process_sobel_magnitude_of_the_gradient(all_variables_in_pipeline)

    # Step 7
    my_image_processor_util.combining_all_thresholds(all_variables_in_pipeline)

    # Step 8
    my_image_processor_util.perspective_transofrm_of_the_combined_binary_image(all_variables_in_pipeline)

    # Step 9
    finding_lanes.finding_lane_pixels(all_variables_in_pipeline)
    
    # Step 10
    finding_lanes.generate_polynomial_plot_points(all_variables_in_pipeline)

    # Step 11
    finding_lanes.measure_curvature(all_variables_in_pipeline)

    # Step 12
    finding_lanes.measure_vehicle_offset(all_variables_in_pipeline)
    
    # Step 13
    plot_my_images.draw_overlay(all_variables_in_pipeline)
    
    # Step 14
    if(singleImageExecution):
        plot_my_images.draw_sliding_windows(all_variables_in_pipeline)

    return all_variables_in_pipeline