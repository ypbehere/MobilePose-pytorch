from tqdm import tqdm
import numpy as np
import cv2
import math

import argparse

import torch
from dataloader_new import ImageDataset, VideoDataset, DataWriter, prepare_cutting_dataloaders
from network import CoordRegressionNetwork

from torch.utils.data import DataLoader


def display_pose(img, pose, var):
    # mean=np.array([0.485, 0.456, 0.406])
    # std=np.array([0.229, 0.224, 0.225])
    pose = pose.data.cpu().numpy()
    img = img.cpu().numpy()  # .transpose(1, 2, 0)
    color = (0, 255, 255)
    pairs = [[8, 9], [11, 12], [11, 10], [2, 1], [1, 0],
             [13, 14], [14, 15], [3, 4], [4, 5], [8, 7], [7, 6], [6, 2], [6, 3], [8, 12], [8, 13]]
    # img = np.clip(img*std+mean, 0.0, 1.0)
    img_width, img_height, _ = img.shape
    pose[:, 0] *= 2
    pose = ((pose + 1) * np.array([img_height, img_width])-1) / 2  # pose ~ [-1,1]

    part_line = {}
    for n in range(pose.shape[0]):
        if var[n][0] > 0.01 or var[n][1] > 0.01:
            continue
        cor_x, cor_y = int(pose[n, 0]), int(pose[n, 1])
        part_line[n] = (int(cor_x), int(cor_y))
        bg = img.copy()
        cv2.circle(bg, (int(cor_x), int(cor_y)), 5, color, -1)
        # Now create a mask of logo and create its inverse mask also
        transparency = 0.9
        img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)

    # for i, (start_p, end_p) in enumerate(pairs):
    #     if start_p in part_line and end_p in part_line:
    #         start_xy = part_line[start_p]
    #         end_xy = part_line[end_p]
    #         bg = img.copy()

    #         X = (start_xy[0], end_xy[0])
    #         Y = (start_xy[1], end_xy[1])
    #         mX = np.mean(X)
    #         mY = np.mean(Y)
    #         length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
    #         angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
    #         stickwidth = 3
    #         polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
    #         cv2.fillConvexPoly(bg, polygon, color)
    #         # cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
    #         transparency = 0.9
    #         img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def do_detect(test_data, model):
        """
        Example:
        eval_coco('/home/yuliang/code/PoseFlow/checkpoint140.t7',
        'result-gt-json.txt', 'result-pred-json.txt')
        """

        image, origin_image = test_data

        with torch.no_grad():
            coords, heatmaps, var = model(image)

        result = display_pose(origin_image[0], coords[0], var[0])

        return (result, coords[0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--model', type=str, required=True, default='')
    parser.add_argument('--t7', type=str, required=True, default='')
    parser.add_argument('--test_dir', type=str, required=True, default='')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--save_path', type=str, default='./result')
    # parser.add_argument('--fourcc', type=str, default='XVID')
    args = parser.parse_args()

    modelpath = args.t7

    device = torch.device("cpu")

    input_size = 224
    modelname = args.model

    test_dataset = VideoDataset(args.test_dir)
    fps, frame_size = test_dataset.video_info()
    print(fps, frame_size)
    data_writer = DataWriter(args.save_video, args.save_path, fps=fps, frame_size=frame_size)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = CoordRegressionNetwork(n_locations=16, backbone=modelname)
    net.load_state_dict(torch.load(modelpath, map_location='cpu'))
    net = net.eval()
    # get all test data
    all_test_data = {}
    for i_batch, data in enumerate(tqdm(test_dataloader)):
        result, coords = do_detect(data, net)
        data_writer.save(result, coords)

    data_writer.stop()
