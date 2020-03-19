from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import os.path
from os import path


def trim_a_video(input_video_file, target_video_file, start_time, end_time):
    ffmpeg_extract_subclip(input_video_file, start_time, end_time, targetname=target_video_file)


def divide_video_into_frames(input_video_file, output_directory_path):
    video = cv2.VideoCapture(input_video_file)
    num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print(num_of_frames)

    i = 0
    while video.isOpened():
        ret, frame = video.read()
        final_output_path = output_directory_path + "/" + str(i+1) + ".jpg"
        if ret:
            cv2.imwrite(final_output_path, frame)
        i = i + 1
        if (i > (num_of_frames-1)):
            video.release()
            print("Video is divided into frames completed.")

if __name__ == "__main__":
    input_video_file = "project_video.mp4"
    trimmed_down_video_file = "project_video_trim_down.mp4"

    if not path.exists(trimmed_down_video_file):
        trim_a_video(input_video_file, trimmed_down_video_file, 22, 25)

    output_dir_path = "failed_frames_auto_generated"
    divide_video_into_frames(trimmed_down_video_file, output_dir_path)
    
    