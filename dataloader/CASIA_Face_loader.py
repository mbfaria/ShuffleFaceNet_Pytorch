import numpy as np
import imageio
import os
from sklearn import preprocessing
import torch
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("..")

class CASIA_Face(object):
    def __init__(self, root):
        self.image_list = []
        self.label_list = []

        for r, _, files in os.walk(root):
            for f in files:
                self.image_list.append(os.path.join(r, f))
                self.label_list.append(os.path.basename(r))

        le = preprocessing.LabelEncoder()
        self.label_list = le.fit_transform(self.label_list)
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = imageio.imread(img_path)
        #img = np.resize(img, (112, 112))


        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        flip = np.random.choice(2)*2-1
        img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    data_dir = '/home/users/matheusb/recfaces/datasets/CASIA-WebFace/'
    dataset = CASIA_Face(root=data_dir)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)
