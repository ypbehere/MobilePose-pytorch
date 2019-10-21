import os
import json
import torch
import argparse
import numpy as np

from network import CoordRegressionNetwork
from dataloader_new import prepare_cutting_dataloaders


pose_map = {
            0: 16,  # left foot
            1: 14,  # left knee
            2: 12,  # left hip
            3: 11,  # right hip
            4: 13,  # right knee
            5: 15,  # right foot
            10: 10,  # left hand
            11: 8,  # left elbow
            12: 6,  # left shoulder
            13: 5,  # right shoulder
            14: 7,  # right elbow
            15: 9  # right hand
}


def cosine_dist(a, b):
    if len(a) == 0:
        return 0
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = num / denom
    return cos


def calc_similarity(test_dataloaders, net, save_path, video_name):
    file = open(os.path.join(save_path, video_name.split('.')[0]+'.txt'), 'w')
    for index, test_dataloader in enumerate(test_dataloaders):
        index_result = []
        for i_batch, data in enumerate(test_dataloader):
            image, origin_image = data
            with torch.no_grad():
                coords, var = net(image)
                index_result.append((coords[0].numpy(), var[0].numpy()))
                index_result.append((coords[1].numpy(), var[1].numpy()))
        index_template = json.load(open('templates/alphapose-results-{}.json'.format(index+1)))
        assert len(index_result) == len(index_template) // 5, 'ERROR: Not aligned with template'

        total_dist = 0
        total_cnt = 0
        for i, (result_coords, var) in enumerate(index_result):
            if var[12].sum() > 0.02 or var[13].sum() > 0.02:
                continue
            result_center = (result_coords[12] + result_coords[13]) / 2
            result_coords -= result_center
            result_upper = []
            result_upper_flags = []
            result_inferior = []
            result_inferior_flags = []
            for j in range(10, 16):
                if var[j].sum() < 0.02:
                    result_upper.append(result_coords[j])
                    result_upper_flags.append(j)
            for j in range(6):
                if var[j].sum() < 0.02:
                    result_inferior.append(result_coords[j])
                    result_inferior_flags.append(j)

            template_coords = np.array(index_template[i * 5]['keypoints'])
            template_coords = template_coords.reshape(-1, 3)[:, :2]
            template_coords /= [720, 1280]
            template_coords *= 2
            template_coords -= 1
            template_center = (template_coords[5] + template_coords[6]) / 2
            template_coords -= template_center
            template_upper = []
            template_inferior = []
            for j in result_upper_flags:
                template_upper.append(template_coords[pose_map[j]])
            for j in result_inferior_flags:
                template_inferior.append(template_coords[pose_map[j]])

            result_upper = np.array(result_upper).reshape((-1,))
            result_inferior = np.array(result_inferior).reshape((-1,))
            template_upper = np.array(template_upper).reshape((-1,))
            template_inferior = np.array(template_inferior).reshape((-1,))

            assert len(result_upper) == len(template_upper), 'ERROR: Upper pose not aligned'
            assert len(result_inferior) == len(template_inferior), 'ERROR: Inferior pose not aligned'

            dist = cosine_dist(result_upper, template_upper) + cosine_dist(result_inferior, template_inferior) * 0.5
            total_dist += dist
            total_cnt += 1

        total_dist /= total_cnt
        print('Chap. {} Similarity: {}'.format(index, total_dist))
        file.write('Chap. {} Similarity: {}\n'.format(index, total_dist))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--t7', type=str, default='models/resnet18_224_adam_best.t7')
    parser.add_argument('--test_dir', type=str, required=True, default='')
    parser.add_argument('--save_path', type=str, default='./result')
    args = parser.parse_args()

    modelpath = args.t7
    device = torch.device("cpu")
    modelname = args.model

    net = CoordRegressionNetwork(n_locations=16, backbone=modelname)
    net.load_state_dict(torch.load(modelpath, map_location='cpu'))
    net = net.eval()

    os.makedirs(args.save_path, exist_ok=True)
    videos = os.listdir(args.test_dir)
    print('Info: {} videos in total'.format(len(videos)))
    for video in videos:
        print('Info: {} detecting start!'.format(video))
        test_dataloaders, fps, frame_size = prepare_cutting_dataloaders(os.path.join(args.test_dir, video))
        calc_similarity(test_dataloaders, net, args.save_path, video)
