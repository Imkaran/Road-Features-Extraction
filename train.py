import os
import random
from datasets import RoadDataset
from utils.augmentation import get_training_augmentation, get_preprocessing, get_validation_augmentation
from segmentation_model import SegmentationModel
from config import *
from torch.utils.data import DataLoader
import torch
from shutil import move
from tqdm import tqdm
from torch import nn

os.makedirs('weights', exist_ok=True)

def train_valid_split():
    print("Split dataset to train and val set")
    i_images = os.listdir(os.path.join(DATA_DIR, 'train', 'input'))
    mask = os.listdir(os.path.join(DATA_DIR, 'train', 'output'))
    i_images_set = set(i_images)
    images = list(i_images_set.intersection(mask))

    random.shuffle(images)
    val_split = int(len(images) * 0.2)

    dst_input = os.path.join(DATA_DIR, 'valid', 'input')
    dst_output = os.path.join(DATA_DIR, 'valid', 'output')
    os.makedirs(dst_input, exist_ok=True)
    os.makedirs(dst_output, exist_ok=True)

    for i in tqdm(range(val_split), desc='val_split'):
        input_image = os.path.join(DATA_DIR, 'train', 'input', images[i])
        output_image = os.path.join(DATA_DIR, 'train', 'output', images[i])

        move(input_image, dst_input)
        move(output_image, dst_output)


def run():
    train_epoch, valid_epoch = seg_model.train_valid_epochs()
    best_score = 0

    for i in range(0, EPOCHS):

        print('\nEpoch: {}/{}'.format(i+1, EPOCHS))
        _ = train_epoch.run(train_loader)
        
        valid_logs = valid_epoch.run(valid_loader)

        # saving best weight
        if best_score < valid_logs['iou_score']:
            best_score = valid_logs['iou_score']
            torch.save(model, './weights/road_best_ckpt.pth')
            print('Best Model saved!')

    torch.save(model, './weights/road_last_ckpt.pth')

if __name__ == '__main__':
    if not os.path.exists(os.path.join(DATA_DIR, 'valid')):
        train_valid_split()

    x_train_dir = os.path.join(DATA_DIR, 'train', 'input')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'output')

    x_valid_dir = os.path.join(DATA_DIR, 'valid', 'input')
    y_valid_dir = os.path.join(DATA_DIR, 'valid', 'output')

    seg_model = SegmentationModel(ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION, device=DEVICE)
    model = seg_model.unet()

    print(model)
    preprocessing_fn = seg_model.preprocess()

    train_dataset = RoadDataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        net_input=(640, 640)
    )

    valid_dataset = RoadDataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        net_input=(640, 640)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)

    run()
    del model