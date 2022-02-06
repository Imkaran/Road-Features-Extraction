from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np
from utils.utils import resize_pad

class RoadDataset(BaseDataset):
    """Road Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    CLASSES = ['non_road', 'road']

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None, net_input=(640, 640)):

        set1 = set(os.listdir(images_dir))
        set2 = os.listdir(masks_dir)
        image_list = list(set1.intersection(set2))

        self.ids = image_list
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.net_input = net_input

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        image, mask = resize_pad(image, mask, target_dim=self.net_input)
        
        mask = np.where(mask==255, 1, mask)
        masks = [(mask == v) for v in self.class_values]

        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

