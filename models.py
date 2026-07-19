import torch
import torch.nn as nn
import torchvision.models as models

class CinematicAestheticClassifier(nn.Module):
    """
    ResNet-based classifier to identify the cinematic era/style of an image.
    """
    def __init__(self, num_classes=3):
        super(CinematicAestheticClassifier, self).__init__()
        # Using ResNet18 as an efficient backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace the final fully connected layer to match our era count
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class VGGStyleExtractor(nn.Module):
    """
    VGG19 Feature Extractor for Neural Style Transfer.
    Extracts intermediate layers representing content and style.
    """
    def __init__(self):
        super(VGGStyleExtractor, self).__init__()
        # Load pre-trained VGG19 features
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # We only need specific layers for style and content mapping
        self.chosen_features = ['0', '5', '10', '19', '28'] # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        self.features = nn.ModuleList([vgg[i] for i in range(29)])

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.features):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features