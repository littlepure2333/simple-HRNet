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
    """
    Extract keypoints from videos and save the results as json files. 
    The videos should be gathered in one folder named with category_id (e.g. 0).
    Input the folder path as videos_path and output json files in the same path.
    Also a preview picture in the middle frame will be saved in the same path.
    """
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

    print("> work on {}".format(device))

    image_resolution = ast.literal_eval(image_resolution)

    # load videos
    videos_name = [vn for vn in os.listdir(videos_path) if vn.split(".")[1] == "mp4"]

    print("> total {} videos in {}".format(len(videos_name), videos_path))

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
        print("{}. reading video: {}".format(i, video_name))

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
            print("  pose estimating total {} frames".format(T))
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
            print("  saving a preview picture in the middel frame: {}".format(middle))
            preview_frame = frames[middle]
            preview_point = points[middle]
            for i, pt in enumerate(preview_point):
                preview_frame = draw_points_and_skeleton(preview_frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=i,
                                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                                        points_palette_samples=10)
            preview_path = os.path.join(videos_path, '{}_preview.png'.format(video_name.split('.')[0]))
            cv2.imwrite(preview_path, preview_frame)
        
            # encapsulate annotations
            annotations = []
            T, M, V, C = points.shape # points: [T, M, V, C]
            for t in range(T):
                frame = {"frame_index": t}
                person = []
                for m in range(M):
                    skeleton = []
                    for v in range(V):
                        skeleton.append((points[t,m,v,0], points[t,m,v,1], points[t,m,v,2]))
                    keypoint = {"keypoint": skeleton}
                    person.append(keypoint)
                frame["person"] = person
                annotations.append(frame)

            # encapsulate info
            width = video.get(3)
            height= video.get(4)
            info = {
                "video_name": video_name,
                "resolution": (width, height),
                "num_frame": T,
                "num_person": M,
                "num_keypoints": V,
                "keypoint_channels": ("x", "y", "score"),
                "version": 1.0
            }
            print("  " + info)

            # encapsulate category_id
            folder_name = videos_path.split("/")[-1]
            category_id = folder_name
            print("   category_id: {}".format(category_id))

            # encapsulate all
            data_dict = {
                "info": info,
                "annotations": annotations,
                "category_id": category_id
            }

            # save json
            data_json = json.dumps(data_dict)
            json_path = os.path.join(videos_path, '{}.json'.format(video_name.split('.')[0]))
            with open(json_path,'w')as f:
                f.write(data_json)
            print("  data json has been saved in {}".format(videos_path))
        print("  total time: {}".format(time.time() - t))
    
    print("the longest frames in these videos is {}".format(longest))
    print("the shortest frames in these videos is {}".format(shortest))


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
    args = parser.parse_args()
    main(**args.__dict__)
