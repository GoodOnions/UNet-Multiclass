import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from utils import get_supermario_data, decode_segmap



class UNET(nn.Module):
    """
    Since the original UNET paper was designed for medical image segmentation, we need to adapt it to our problem.
    We will use the same architecture but we will change the number of input channels and output classes.
    we also reduce the number of layers to reduce the computational cost and model complexity.
    From the original paper:
        layers = [in_channels, 64, 128, 256, 512, 1024]
    We will use:
        layers = [in_channels, 32, 64, 128, 256, 512]
    """
    
    def __init__(self, in_channels=3, classes=1, layers=[32, 64, 128, 256, 512]):
        super(UNET, self).__init__()
        layers.insert(0, in_channels)
        self.layers = layers
        
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])

        for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1]):
            print("Down",layer, layer_n)
            
        self.double_conv_ups = nn.ModuleList(
        [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])

        for layer in self.layers[::-1][:-2]:
            print("Up", layer)
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(self.layers[1], classes, kernel_size=1)

        
    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
    
    def forward(self, x):
        # down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        
        # up layers
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

            # Stop here for a middle result
            #
            
        x = self.final_conv(x)
        
        return x

    def predict(self, frame):

        img = transforms.ToTensor()(frame).to(self.device)
        img = img.unsqueeze(0)
        prediction = self.forward(img)
        prediction = torch.nn.functional.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, dim=1).squeeze()
        prediction = prediction.float().detach().cpu().numpy()
        segm_rgb = decode_segmap(prediction)
        return segm_rgb



