import os
import sys
import argparse
import ast
import cv2
import time
import torch
import numpy as np
import json

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict


def main(videos_path, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution, 
         yolo_model_def, yolo_weights_path, single_person, max_batch_size, device, same_frame, label):
    # init environment
    # TODO multi GPU
    shortest = sys.maxsize
    longest = 0
    label_dict = {}
    label_dict_filled = {}
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available() and True:
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    print("work on {}".format(device))

    image_resolution = ast.literal_eval(image_resolution)

    # load videos
    videos_name = [vn for vn in os.listdir(videos_path) if vn.split(".")[1] == "mp4"]

    print("total {} videos in {}".format(len(videos_name), videos_path))

    # load model
    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        resolution=image_resolution,
        multiperson=not single_person,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_weights_path=yolo_weights_path,
        device=device
    )

    # read videos
    for i, video_name in enumerate(videos_name):
        # read video
        filename = os.path.join(videos_path,video_name)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
        print("{}.read video: {}".format(i, video_name))

        # read frames
        frames = []
        while True:
            t = time.time()

            ret, frame = video.read()  # read(): Grabs, decodes and returns the next video frame.
            if not ret:
                break
            
            frames.append(frame)

        # pose estimation
        if len(frames) > 0:
            T = len(frames)
            if T > longest:
                longest = T
            if T < shortest:
                shortest = T
            print("\ttotal {} frames".format(T))
            frames = np.array(frames)
            # frames: a stack of n images with shape=(n, height, width, BGR color channel)
            # points: list of n np.ndarrays with shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).
            points = model.predict(frames)
            max_M = 0
            for i in range(len(points)):
                if points[i].shape[0] > max_M:
                    max_M = points[i].shape[0]

            # save a preview picture in the middle frame of the video
            middle = int(len(frames)/2)
            print("\tmiddel: {}".format(middle))
            preview_frame = frames[middle]
            preview_point = points[middle]
            for i, pt in enumerate(preview_point):
                preview_frame = draw_points_and_skeleton(preview_frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=i,
                                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                                        points_palette_samples=10)
            preview_path = os.path.join(videos_path, '{}_preview.png'.format(video_name.split('.')[0]))
            cv2.imwrite(preview_path, preview_frame)
        
            # points: [T, M, V, C]
            points_17 = np.zeros((T, max_M, 17, 3))
            for i, p in enumerate(points):
                M = p.shape[0]
                if M > 0:
                    points_17[i, 0:M, :, :] = p
            print("\tpoints_17.shape: {}".format(points_17.shape))
            # 17->18 points
            if points_17.shape[1] > 0:
                neck = 0.5 * (points_17[:, :, 5, :] + points_17[:, :, 6, :])  # neck is the mean of shoulders
                neck = np.expand_dims(neck, 2)  # shape: [T, M, 3] -> [T, M, 1, 3]
                points_18 = np.concatenate((points_17, neck), 2)
                # convert coco format to openpose format
                points = points_18[:, :, [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3], :]

            # save npy
            print("\tpoints_18.shape: {}".format(points.shape))
            save_path = os.path.join(videos_path, '{}.npy'.format(video_name.split('.')[0]))
            np.save(save_path, points)

            # add label
            if label is not None:
                video_key = video_name.split('.')[0]
                label_dict[video_key] = label
                video_key_filled = "filled_" + video_name.split('.')[0]
                label_dict_filled[video_key_filled] = label

        # save label json
        if label is not None:
            label_json = json.dumps(label_dict)
            json_path = os.path.join(videos_path, '{}.json'.format(label))
            label_json_filled = json.dumps(label_dict_filled)
            json_filled_path = os.path.join(videos_path, '{}.json'.format(str(label)+"_filled"))
            with open(json_path,'w')as f:
                f.write(label_json)
            with open(json_filled_path,'w')as f:
                f.write(label_json_filled)
            print("\tlabel json has saved in {}".format(videos_path))
        print("\tnpy has saved in {}".format(videos_path))
        print("\ttotal time: {}".format(time.time() - t))
    
    print("the longest frames in these videos is {}".format(longest))
    print("the shortest frames in these videos is {}".format(shortest))

    if same_frame is not None:
        if same_frame < longest:
            print("can't fill up the same number of frames because should longer then longest frames")
        else:
            fill_same_frame(videos_path, same_frame)
    else:
        fill_same_frame(videos_path, longest)

def fill_same_frame(npys_path, max_frame):
    # load npys
    npys_name = [nn for nn in os.listdir(npys_path) if nn.split(".")[1] == "npy"]
    for i, npy_name in enumerate(npys_name):
        # read npy
        filename = os.path.join(npys_path, npy_name)
        data = np.load(filename)  # T, M, V, C
        data_filled = data
        data_fill_index = 0
        # replay frames until the number of frames is max_frame
        while data_filled.shape[0] < max_frame:
            frame_fill = np.expand_dims(data_filled[data_fill_index], 0)
            data_filled = np.concatenate((data_filled, frame_fill), axis = 0)
            data_fill_index = data_fill_index + 1
        print("{}.filled_{} shape is {}".format(i, npy_name, data_filled.shape))
        # save the npy file
        save_path = os.path.join(npys_path, 'filled_{}.npy'.format(npy_name.split('.')[0]))
        np.save(save_path, data_filled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_path", "-v", help="the videos folder path)", type=str, default=None)
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)", type=str, default=None)
    parser.add_argument("--yolo_model_def", help="path to yolo model definition file",
                        type=str, default="./models/detectors/yolo/config/yolov3.cfg")
    parser.add_argument("--yolo_weights_path", help="path to yolo pretrained weights file",
                        type=str, default="./models/detectors/yolo/weights/yolov3.weights")
    parser.add_argument("--same_frame", help="auto fill up to same number of frames", type=int, default=None)
    parser.add_argument("--label", help="the class label", type=int, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
