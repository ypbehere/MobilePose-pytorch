import os
import cv2
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset


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
