# Road-Features-Extraction
Road segmentation of satellite imagery

## Install Requirements
```pip3 install requirements.txt```

## Folder Structure
```
data/Road
├── train
|    └── input
|    └── output
├── valid
|   └── input
|   └── output
└── test
|   └── input
|   └── output
```

## Update config.py
```
DATA_DIR = './data/Road/' #dataset directory
ENCODER = 'efficientnet-b6'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['road']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda' # 'cuda' or 'cpu'
gpu_ids = '1'

BATCH_SIZE = 8
EPOCHS = 20
workers = os.cpu_count()
```

## Train
``` python3 train.py ```

## Inference
``` python3 test.py ./data/Road/test/input ./data/Road/test/output ./weights/road_best_ckpt.pth ./result ```
