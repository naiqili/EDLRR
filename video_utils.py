import numpy as np
import os
import cv2

def video_to_numpy_array(video_path):

    rgb_frame_list = []
    video_read_capture = cv2.VideoCapture(video_path)
    while video_read_capture.isOpened():
        result, frame = video_read_capture.read()
        if not result:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame_list.append(frame)

    video_read_capture.release()

    video_nparray = np.array(rgb_frame_list)

    return video_nparray

def numpy_array_to_video(numpy_array,video_out_path):
    video_height = numpy_array.shape[1]
    video_width = numpy_array.shape[2]

    out_video_size = (video_width,video_height)
    output_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))
    video_write_capture = cv2.VideoWriter(video_out_path, output_video_fourcc, 15, out_video_size)

    for frame in numpy_array:
        video_write_capture.write(frame)

    video_write_capture.release()

def save_video(video_np_array, w, root, out_name, residual=False, use_f=False):
    shape = video_np_array[..., 0].shape
    w = w.reshape(shape)
    if use_f:
        w = w * 255.0
    if residual:
        w = np.abs(w)
    w = np.clip(w,0,255)
    video_np_array[..., 0] = w
    video_np_array[..., 1] = w
    video_np_array[..., 2] = w
    save_path = os.path.join(root, out_name)
    numpy_array_to_video(video_np_array,save_path)
    return