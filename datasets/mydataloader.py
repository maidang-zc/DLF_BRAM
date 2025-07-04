import numpy as np
from collections import OrderedDict
from collections import Counter
import os
import glob
import cv2
import csv
import torchaudio
import torch
import torch.utils.data as data
import torchvision.transforms as T
import json
rng = np.random.RandomState(2020)
import random
random.seed(2020)
import random
seed = 1111
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

import natsort
from utils.train_split import (
    train_test_split_bg, train_test_split_MU3d, 
    train_test_split_Dolos
)
from utils.image_crop import np_load_frame_5, np_load_frame_7

class Deception_Dataset(data.Dataset):

    def __init__(self, video_folder, annotations_folder,  data_name, train_flag, img_size, frame_len, blocks, train = True, Cross_val = False):
        """
        Unified data loading class for multiple datasets, supporting key block-level image region extraction.

        Args:
            video_root: Root directory containing video frames (each video in a separate subfolder)
            anno_root: Directory containing key region annotations (.json files)
            data_name: Dataset name: Bag, MU3D, Dolos, RealLife
            train_flag: Dataset split ID (1 to 3)
            img_size: Size to which each block image is resized
            frame_len: Number of frames per sample
            blocks: Number of blocks (5 or 7)
            train: Whether the dataset is used for training
            Cross_val: Whether to use the entire dataset (cross-validation mode)
        """

        self.dir = video_folder
        self.annotations_folder = annotations_folder
        self.data_name = data_name
        self.train_flag = train_flag  # train_test_split set 1，2，3
        self.train = train
        self.img_size = img_size
        self.frame_len = frame_len
        self.blocks = blocks
        self.cross_val = Cross_val
        self.videos = OrderedDict()

        self.setup()
        self.samples = self.get_all_samples()


    @staticmethod
    def get_videos(data_name, dir, train_flag, train, Cross_val):
        if data_name == 'Bag':
            trainlist, testlist = train_test_split_bg(dir, train_flag)
        elif data_name == 'MU3D':
            trainlist, testlist = train_test_split_MU3d(dir, train_flag)
        elif data_name == 'Dolos':
            trainlist, testlist = train_test_split_Dolos(dir, train_flag)
            trainlist = list(trainlist)
            testlist = list(testlist)
        else:
            raise ValueError("Invalid data_name. Expected 'BOFL' or 'MU3D' or 'RealLife' of 'Dolos'.")

        if train:
            videos = [os.path.join(dir, video) for video in trainlist]
        elif Cross_val:
            videos = [os.path.join(dir, video) for video in trainlist + testlist]
        else:
            videos = [os.path.join(dir, video) for video in testlist]
        return videos

    def setup(self):
        self.video_path = self.get_videos(self.data_name, self.dir, self.train_flag, self.train, self.cross_val)

        for video in sorted(self.video_path):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            faces = glob.glob(os.path.join(video, '*.jpg'))

            annotation_file = os.path.join(self.annotations_folder,video_name+".json")
            with open(annotation_file) as json_file:
                #  load alphapose data from json file
                data = json.load(json_file)
            vid = []
            img = []
            for face in faces:
                img.append(face.split('/')[-1])
            for key, value in data.items():
                vid.append(key)

            vid = set(vid)
            img = set(img)
            facesnew = list(img & vid)
            facesnew = natsort.natsorted(facesnew)

            self.videos[video_name]['frame'] = [os.path.join(video, i) for i in facesnew]
            self.videos[video_name]['length'] = len(facesnew)

            json_file.close()
    def get_all_samples(self):
        frames = []

        for video in sorted(self.video_path):
            video_name = video.split('/')[-1]
            frames.extend(self.videos[video_name]['frame'][0::self.frame_len][:-1])

        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        if "lie" in video_name:
            label = 1
        elif "truth" in video_name:
            label = 0
        else:
            print("error label")
        idx = self.videos[video_name]['frame'].index(self.samples[index])
        annotation_file = os.path.join(self.annotations_folder,video_name+".json")
        with open(annotation_file) as json_file:
            #  load alphapose data from json file
            pic = json.load(json_file)

        batch = []

        for i in range(self.frame_len):
            frame = self.videos[video_name]['frame'][i+idx]
            frame_name = frame.split('/')[-1]
            try:
                annotation = pic[frame_name]
            except KeyError:
                print(annotation_file)
                print(f"KeyError: '{frame_name}' not found in pic dictionary.")
                raise ValueError(f"Frame name '{frame_name}' not found in pic dictionary.")
            # annotation  = pic[frame_name]
            if self.blocks == 7:
                batch.append(np_load_frame_7(frame,annotation,self.img_size))
            elif self.blocks == 5:
                batch.append(np_load_frame_5(frame,annotation,self.img_size))

        batch = np.asarray(batch).astype(np.float32) # frame_len,blocks,height,width,channels
        json_file.close()

        return batch.transpose(4, 0, 1, 2, 3),label,video_name # (channels,frame_len,blocks,height,width)

    def __len__(self):
        return len(self.samples)

