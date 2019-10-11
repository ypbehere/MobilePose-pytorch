import os
import cv2
import torch
import numpy as np
from skimage import io
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)
        self.image_num = len(self.image_list)
        self.transform = transform
        self.output_size = (224, 224)

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        origin_image = io.imread(os.path.join(self.root_dir, self.image_list[idx]))

        image = origin_image/256.0
        h, w = image.shape[:2]
        im_scale = min(float(self.output_size[0]) / float(h), float(self.output_size[1]) / float(w))
        new_h = int(image.shape[0] * im_scale)
        new_w = int(image.shape[1] * im_scale)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        # left_pad = (self.output_size[1] - new_w) // 2
        # right_pad = (self.output_size[1] - new_w) - left_pad
        # top_pad = (self.output_size[0] - new_h) // 2
        # bottom_pad = (self.output_size[0] - new_h) - top_pad
        # mean = np.array([0.485, 0.456, 0.406])
        # pad = ((top_pad, bottom_pad), (left_pad, right_pad))
        # image = np.stack([np.pad(image[:, :, c], pad, mode='constant', constant_values=mean[c])
        #                  for c in range(3)], axis=2)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image[:, :, :3] = (image[:, :, :3]-mean)/(std)
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()

        return (image, origin_image)


class VideoDataset(Dataset):
    def __init__(self, video_dir):
        super(VideoDataset, self).__init__()
        self.video_dir = video_dir
        stream = cv2.VideoCapture(self.video_dir)
        assert stream.isOpened(), 'ERROR: Cannot capture source'
        self.fps = stream.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.data_len = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_size = (256, 256)

        self.frames = []
        for i in range(self.data_len):
            _, frame = stream.read()
            self.frames.append(frame)

        print('Finish Loading, {} frames in total'.format(self.data_len))

    def __len__(self):
        return self.data_len // 5

    def __getitem__(self, idx):
        origin_image = self.frames[idx * 5]
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

        image = origin_image/256.0
        h, w = image.shape[:2]
        im_scale = min(float(self.output_size[0]) / float(h), float(self.output_size[1]) / float(w))
        new_h = int(image.shape[0] * im_scale)
        new_w = int(image.shape[1] * im_scale)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        left_pad = (self.output_size[1] - new_w) // 2
        right_pad = (self.output_size[1] - new_w) - left_pad
        top_pad = (self.output_size[0] - new_h) // 2
        bottom_pad = (self.output_size[0] - new_h) - top_pad
        mean = np.array([0.485, 0.456, 0.406])
        pad = ((top_pad, bottom_pad), (left_pad, right_pad))
        image = np.stack([np.pad(image[:, :, c], pad, mode='constant', constant_values=mean[c])
                         for c in range(3)], axis=2)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image[:, :, :3] = (image[:, :, :3]-mean)/(std)
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()

        return (image, origin_image)

    def video_info(self):
        return (self.fps, self.frame_size)


cut_seconds = [(0, 5),  # 1_1
               (40, 45),  # 1_2
               (110, 125),  # 2_1
               (210, 225),  # 3_1
               (330, 335),  # 4_1
               (345, 350),  # 4_2
               (430, 440),  # 5_1
               (450, 455),  # 5_2
               (550, 555),  # 6_1
               (585, 590),  # 6_2
               (675, 685),  # 7_1
               (695, 705),  # 7_2
               (765, 770),  # 8_1
               (800, 805)]  # 8_2


def prepare_cutting_dataloaders(video_dir):
    video_dir = video_dir
    stream = cv2.VideoCapture(video_dir)
    assert stream.isOpened(), 'ERROR: Cannot capture source'
    fps = int(stream.get(cv2.CAP_PROP_FPS))
    data_len = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_time = data_len / fps
    assert abs(total_time - 830) < 20, 'ERROR: Video time should be adjusted'

    frame_groups = []
    frame_group = []
    cut_idx = 0
    lower_bound = cut_seconds[cut_idx][0] * fps
    upper_bound = cut_seconds[cut_idx][1] * fps
    print('Cutting', cut_idx, 'Started')
    for i in range(data_len):
        success, frame = stream.read()
        if i >= lower_bound and i % 5 == 0:
            frame_group.append(frame)
        if i >= upper_bound:
            frame_groups.append(frame_group)
            frame_group = []
            cut_idx += 1
            if cut_idx == 14:
                print('Cutting finished')
                break

            lower_bound = cut_seconds[cut_idx][0] * fps
            upper_bound = cut_seconds[cut_idx][1] * fps
            print('Cutting', cut_idx, 'Started')

    stream.release()

    frame_dataloaders = []
    total_frames = 0
    for frame_group in frame_groups:
        frame_dataset = VideoCuttingDataset(frame_group)
        total_frames += frame_dataset.num_frames
        frame_dataloader = DataLoader(frame_dataset, batch_size=1, shuffle=False)
        frame_dataloaders.append(frame_dataloader)

    print('Finish Loading, {} frames in total'.format(total_frames))

    return frame_dataloaders, fps, frame_size


class VideoCuttingDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.num_frames = len(frames)
        self.output_size = (256, 256)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        origin_image = self.frames[idx]
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

        image = origin_image/256.0
        h, w = image.shape[:2]
        im_scale = min(float(self.output_size[0]) / float(h), float(self.output_size[1]) / float(w))
        new_h = int(image.shape[0] * im_scale)
        new_w = int(image.shape[1] * im_scale)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        left_pad = (self.output_size[1] - new_w) // 2
        right_pad = (self.output_size[1] - new_w) - left_pad
        top_pad = (self.output_size[0] - new_h) // 2
        bottom_pad = (self.output_size[0] - new_h) - top_pad
        mean = np.array([0.485, 0.456, 0.406])
        pad = ((top_pad, bottom_pad), (left_pad, right_pad))
        image = np.stack([np.pad(image[:, :, c], pad, mode='constant', constant_values=mean[c])
                         for c in range(3)], axis=2)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image[:, :, :3] = (image[:, :, :3]-mean)/(std)
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()

        return (image, origin_image)


class DataWriter(object):
    def __init__(self, save_video=False, save_path='./result',
                 fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frame_size=(256, 256)):
        if save_video:
            save_path = os.path.join(save_path, 'res.avi')
            self.stream = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
            assert self.stream.isOpened(), 'ERROR: Cannot open video for writing'

        self.save_video = save_video
        self.final_result = []

    def save(self, image, coords):
        self.stream.write(image)
        self.final_result.append(coords.cpu().numpy())

    def stop(self):
        self.stream.release()
        # print(self.final_result)
