from moviepy.editor import VideoFileClip

# My classes imports
from _CameraCalibration import CameraCalibration
from _MainPipeline import my_pipeline
from _PlotMyImages import PlotMyImages as plot_my_images

def process_each_frame(frame):
    camera_calibration = CameraCalibration.get_calibrated_camera()
    all_variables_in_pipeline = my_pipeline(frame, "", camera_calibration)
    return all_variables_in_pipeline.img_processed

camera_calibration = CameraCalibration.get_calibrated_camera()
input_video_file = "project_video.mp4"
video = VideoFileClip(input_video_file)#.subclip(22,26)
project_video = video.fl_image(process_each_frame)
project_video.write_videofile("generated_output_video_v3.mp4", audio=False)
print("Video saved to: generated_output_video_v3.mp4")