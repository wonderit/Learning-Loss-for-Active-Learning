import numpy as np
import pandas as pd
import pickle
import os

import torch.utils.data
from PIL import Image
from typing import Any, Callable, Optional, Tuple

from torch.utils.data import DataLoader


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict['data'], dict['labels']


def compress_image(prev_image, n):
    height = prev_image.shape[0] // n
    width = prev_image.shape[1] // n
    new_image = np.zeros((height, width), dtype="uint8")
    for i in range(0, height):
        for j in range(0, width):
            new_image[i, j] = prev_image[n * i, n * j]

    return new_image

class CEMDataset(torch.utils.data.Dataset):
    DATASETS_TRAIN = [
        'binary_501',
        'binary_502',
        'binary_503',
        'binary_504',
        'binary_505',
        'binary_506',
        'binary_507',
        'binary_508',
        'binary_509',
        'binary_510',
        'binary_511',
        'binary_512',
        'binary_1001',
        'binary_1002',
        'binary_1003',
        # 'binary_rl_fix_501',
        # 'binary_rl_fix_502',
        # 'binary_rl_fix_503',
        # 'binary_rl_fix_504',
        # 'binary_rl_fix_505',
        # 'binary_rl_fix_506',
        # 'binary_rl_fix_507',
        # 'binary_rl_fix_508',
        # 'binary_rl_fix_509',
        # 'binary_rl_fix_510',
        # 'binary_rl_fix_511',
        # 'binary_rl_fix_512',
        # 'binary_rl_fix_513',
        # 'binary_rl_fix_514',
        # 'binary_rl_fix_515',
        # 'binary_rl_fix_516',
        # 'binary_rl_fix_517',
        # 'binary_rl_fix_518',
        # 'binary_rl_fix_519',
        # 'binary_rl_fix_520',
        # 'binary_rl_fix_1001',
        # 'binary_rl_fix_1002',
        # 'binary_rl_fix_1003',
        # 'binary_rl_fix_1004',
        # 'binary_rl_fix_1005',
        # 'binary_rl_fix_1006',
        # 'binary_rl_fix_1007',
        # 'binary_rl_fix_1008',
    ]

    DATASETS_VALID = [
        'binary_1004',
        'binary_test_1001',
        'binary_test_1002',
        'binary_rl_fix_1009',
        'binary_rl_fix_1010',
        'binary_rl_fix_1011',
        'binary_rl_fix_1012',
        'binary_rl_fix_1013',
        'binary_rl_fix_test_1001',
    ]

    DATASETS_TEST = [
        'binary_new_test_501',
        'binary_new_test_1501',
        # 'binary_rl_fix_1014',
        # 'binary_rl_fix_1015',
        # 'binary_rl_fix_test_1002',
        # 'binary_rl_fix_test_1003',
        # 'binary_rl_fix_test_1004',
        # 'binary_rl_fix_test_1005',
        'binary_test_1101',
    ]

    def __init__(self,
                 root: str,
                train: bool = True,
                 ) -> None:
        self.train = train
        self.root = root

        # super(CEMDataset, self).__init__(root)

        if self.train:
            DATAPATH = os.path.join(root, 'train')
            DATASETS = self.DATASETS_TRAIN
        else:
            DATAPATH = os.path.join(root, 'test')
            DATASETS = self.DATASETS_TEST

        self.data: Any = []
        self.targets = []

        print('data loading ... ')

        # load Train dataset
        for data in DATASETS:
            dataframe = pd.read_csv(os.path.join(DATAPATH, '{}.csv'.format(data)), delim_whitespace=False, header=None)
            dataset = dataframe.values

            # split into input (X) and output (Y) variables
            fileNames = dataset[:, 0]

            # 1. first try max
            dataset[:, 1:25] /= 2767.1
            self.targets.extend(dataset[:, 1:25])
            for idx, file in enumerate(fileNames):
                try:
                    image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(int(file))))
                    image = np.array(image, dtype=np.uint8)
                except (TypeError, FileNotFoundError) as te:
                    image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(idx + 1)))
                    try:
                        image = np.array(image, dtype=np.uint8)
                    except:
                        continue
                image = compress_image(image, 5)
                self.data.append(np.array(image).flatten(order='C'))
                # self.data.append(np.array(image))

        # print(len(self.data), 'previous')
        # print(np.vstack(self.data).shape, 'previous')
        self.data = np.vstack(self.data).reshape(-1, 1, 20, 40)
        # print(np.vstack(self.data).shape, 'middle')
        self.data = self.data.transpose((0, 1, 2, 3))  # convert to HWC CHW
        # print(self.data.shape, 'after')
        # exit()
        print(f'Data Loading Finished. len : {len(self.data)}')


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":

    batch_size = 1
    data_dir = './maxwellfdfd'

    train_dataset = CEMDataset(data_dir, train=False)
    train_loader = DataLoader(train_dataset, batch_size=32,
                              pin_memory=True)

