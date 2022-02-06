from loguru import logger
import numpy as np
import sys
import cv2

logger.add(sys.stderr, format="{time} {level} {message}", filter="road_segmentation", level="INFO")

def resize_pad(rgb_image, mask_image, target_dim=(1536, 1536)):
    rgb_height, rgb_width, _ = rgb_image.shape
    mask_height, mask_width = mask_image.shape

    if mask_height != rgb_height or mask_width != rgb_width:
        logger.error('rgb image and mask image dimensions are not equal')
        return None, None

    padded_rgb_img = np.zeros((target_dim[0], target_dim[1], 3), dtype=np.uint8)
    padded_mask_img = np.zeros((target_dim[0], target_dim[1]), dtype=np.uint8)
    r = min(target_dim[0] / rgb_height, target_dim[1] / rgb_height)

    resized_rgb_img = cv2.resize(rgb_image, (int(rgb_image.shape[1] * r), int(rgb_image.shape[0] * r)),
                                 interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    resized_mask_img = cv2.resize(mask_image, (int(mask_image.shape[1] * r), int(mask_image.shape[0] * r)),
                                  interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    padded_rgb_img[: int(rgb_image.shape[0] * r), : int(rgb_image.shape[1] * r)] = resized_rgb_img
    padded_mask_img[: int(mask_image.shape[0] * r), : int(mask_image.shape[1] * r)] = resized_mask_img

    return padded_rgb_img, padded_mask_img