import torch
import segmentation_models_pytorch as smp

class SegmentationModel:
    def __init__(self, encoder, encoder_weights, classes, activation, device='cpu'):
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.classes = classes
        self.activation = activation
        self.device = device

    def unet(self):
        self.model = smp.Unet(
            encoder_name=self.encoder,
            encoder_depth=5,
            encoder_weights=self.encoder_weights,
            classes=len(self.classes),
            activation=self.activation
        )
        return self.model

    def preprocess(self):
        return smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)

    def loss_fn(self):
        self.loss = smp.utils.losses.DiceLoss()
        return self.loss

    def metric_fn(self):
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Precision(),
            smp.utils.metrics.Accuracy(),
            smp.utils.metrics.Recall()
        ]

        return self.metrics

    def optimizer_fn(self):
        self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=0.0001)])

    def train_valid_epochs(self):
        self.loss_fn()
        self.metric_fn()
        self.optimizer_fn()
        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )

        return train_epoch, valid_epoch

    def test_epoch(self, best_model):
        self.loss_fn()
        self.metric_fn()
        self.optimizer_fn()

        test_epoch = smp.utils.train.ValidEpoch(
            best_model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )

        return test_epoch