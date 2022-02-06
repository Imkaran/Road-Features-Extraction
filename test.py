import os.path
import sys
from config import *
from datasets import RoadDataset
from utils.augmentation import get_preprocessing, get_validation_augmentation
from segmentation_model import SegmentationModel
from torch.utils.data import DataLoader
import torch
from pprint import pprint
import numpy as np
import cv2
from tqdm import tqdm
from utils.utils import resize_pad

def print_result_metric():
    test_dataset = RoadDataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataloader = DataLoader(test_dataset)

    test_epoch = seg_model.test_epoch(best_model)

    logs = test_epoch.run(test_dataloader)

    print('----------------------------------------------------------------------------')
    pprint(logs)
    print('----------------------------------------------------------------------------')

def save_prediction(image_path, mask_path):
    basename = os.path.basename(image_path)
    raw_image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_path, 0)
    image, _ = resize_pad(raw_image, mask_image, target_dim=(1536, 1536))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessing = get_preprocessing(preprocessing_fn)
    sample = preprocessing(image=image)
    p_image = sample['image']

    x_tensor = torch.from_numpy(p_image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    width_ratio = raw_image.shape[1] / pr_mask.shape[1]
    height_ratio = raw_image.shape[0] / pr_mask.shape[0]

    y_indexes, x_indexes = np.where(pr_mask == 1)
    x_indexes = np.array(x_indexes, dtype='float32')
    y_indexes = np.array(y_indexes, dtype='float32')
    y_indexes *= float(height_ratio)
    x_indexes *= float(width_ratio)

    result = np.full(raw_image.shape, (0, 0, 0), dtype=np.uint8)

    for y, x in tqdm(zip(y_indexes, x_indexes)):
        result[int(y)][int(x)][0] = 0
        result[int(y)][int(x)][1] = 0
        result[int(y)][int(x)][2] = 255

    cv2.imwrite(os.path.join(output_dir, basename), result)

if __name__ == '__main__':
    input_img_dir = sys.argv[1]
    mask_img_dir = sys.argv[2]
    ckpt = sys.argv[3]
    output_dir = sys.argv[4]

    x_test_dir = os.path.join(input_img_dir)
    y_test_dir = os.path.join(mask_img_dir)

    seg_model = SegmentationModel(ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION, device=DEVICE)
    model = seg_model.unet()
    preprocessing_fn = seg_model.preprocess()
    best_model = torch.load(ckpt, map_location=DEVICE)
    best_model.eval()
    print_result_metric()

    for image_name in tqdm(os.listdir(x_test_dir), desc='saving_predictions'):
        img_path = os.path.join(x_test_dir, image_name)
        mask_path = os.path.join(y_test_dir, image_name)
        save_prediction(img_path, mask_path)