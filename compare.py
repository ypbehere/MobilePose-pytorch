import json
import torch
import argparse
from tqdm import tqdm

from network import CoordRegressionNetwork
from dataloader_new import prepare_cutting_dataloaders


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--t7', type=str, default='models/resnet18_224_adam_best.t7')
    parser.add_argument('--video', type=str, required=True, default='')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--save_path', type=str, default='./result')
    args = parser.parse_args()

    modelpath = args.t7

    device = torch.device("cpu")

    input_size = 224
    modelname = args.model

    test_dataloaders, fps, frame_size = prepare_cutting_dataloaders(args.video)

    net = CoordRegressionNetwork(n_locations=16, backbone=modelname)
    net.load_state_dict(torch.load(modelpath, map_location='cpu'))
    net = net.eval()

    for index, test_dataloader in enumerate(test_dataloaders):
        index_result = []
        for i_batch, data in enumerate(tqdm(test_dataloader)):
            image, origin_image = data
            with torch.no_grad():
                coords, _, var = net(image)
                index_result.append((coords, var))
        index_template = json.load(open('templates/alphapose-results-{}.json'.format(index+1)))
        assert len(index_result) == len(index_template), 'ERROR: Not aligned with template'
