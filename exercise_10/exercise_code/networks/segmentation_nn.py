"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.num_classes = num_classes
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.fe = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-3])
        self.cl = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(128,64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(64, 32, stride=1,kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(32, num_classes, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(23),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            nn.Conv2d(num_classes,num_classes,stride=1,kernel_size=3,padding=1),
        )
    
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        output = self.fe(x)
        output = self.cl(output)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output
    
    def general_step(self, batch, batch_idx, mode):
        inputs, targets = batch

        # forward pass
        prediction = self.forward(inputs)

        # loss
        loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(prediction, targets)
        
        return loss
    
    def general_end(self, outputs, mode):
        
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss}

    def validation_end(self, outputs):
        avg_loss = self.general_end(outputs, "val")
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.cl.parameters(), lr=self.hparams["learning_rate"])   
        return optim

    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


    

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
