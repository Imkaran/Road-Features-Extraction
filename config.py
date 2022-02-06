import os

DATA_DIR = './data/Road/'
ENCODER = 'efficientnet-b6'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['road']
ACTIVATION = 'sigmoid' # 'softmax2d' for multiclass segmentation
DEVICE = 'cuda' # 'cuda' or 'cpu'
gpu_ids = '0' # gpu id in string format separated by comma

BATCH_SIZE = 8
EPOCHS = 20
workers = os.cpu_count()
