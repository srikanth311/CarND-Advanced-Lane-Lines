from matplotlib import image as mpimg
from matplotlib import pyplot as plt

# My classes imports
from _CameraCalibration import CameraCalibration
from _MainPipeline import my_pipeline
from _PlotMyImages import PlotMyImages as plot_my_images
import cv2
import sys

camera_calibration = CameraCalibration.get_calibrated_camera()
# Just for testing.
# print(camera_calibration.camera_matrix)

#test_image_path = "test_images/test2.jpg"
file_number=sys.argv[1]
# for file_number in range(1, 75):
all_variables_in_pipeline = None
test_image_path = "failed_frames_auto_generated/" + str(file_number) + ".jpg"
        #test_image_path = "camera_cal/calibration1.jpg"
        #test_image = mpimg.imread("test_images/test1.jpg")
test_image = mpimg.imread(test_image_path)
all_variables_in_pipeline = my_pipeline(test_image, test_image_path, camera_calibration, singleImageExecution=True)

all_processed_images_for_a_given_image = [  #[all_variables_in_pipeline.image, "Input_image"],
                                            #        [all_variables_in_pipeline.gray_image, "Gray_Image", "gray"],
                                            #        [all_variables_in_pipeline.hls_image, "HLS_Image"],
                                            #        [all_variables_in_pipeline.schannel_binary_image, "S-Channel_Binary_Image", "gray"],
                                            #        [all_variables_in_pipeline.undistorted_image, "Undistorted_Image", "gray"],
                                            #        [all_variables_in_pipeline.sobel_binary_image, "Sobel_Binary_Image", "gray"],
                                            #        [all_variables_in_pipeline.sobel_magnitude_binary_image, "Sobel_Magnitude_Image", "gray"],
                                            #        [all_variables_in_pipeline.sobel_direction_binary_image, "Sobel_Direction_Image", "gray"],
                                                    [all_variables_in_pipeline.combined_binary_image, "Combined_Image", "gray"],
                                                    [all_variables_in_pipeline.warped_binary_image, "Warped_Image", "gray"],
                                                    [all_variables_in_pipeline.img_sliding_window, "Sliding_Window_With_Polynomial", "gray"],
                                                    [all_variables_in_pipeline.img_processed, "Final_Image_with_Lanes"],
                                         ]  
#plot_my_images.plot_all_images(all_processed_images_for_a_given_image, 4, 3, "output_images/project2_pipeline_test2_output.png")
plot_my_images.plot_all_images(all_processed_images_for_a_given_image, 2, 2, "failed_images_output/combined_frames/c_"+ str(file_number) + ".png")
    # plot_my_images.plot_all_images(all_processed_images_for_a_given_image, 2, 3, "output_images/project2_color_gradient_test2_output.png")
    # plot_my_images.plot_all_images(all_processed_images_for_a_given_image, 1, 2, "output_images/project2_warped_sliding_test2_output.png")

#cv2.imwrite("failed_images_output/sliding_window/s_"+ str(file_number) + ".png", all_variables_in_pipeline.img_sliding_window)
#plt.imshow(all_variables_in_pipeline.img_sliding_window)
#plt.show()