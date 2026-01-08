
import torch
import torch.nn as nn
import torchvision.models as models
import config

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Load VGG19 pretrained on ImageNet
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # We want features from specific layers. Structure of VGG19 features:
        # Layer indices often used: 3 (relu1_2), 8 (relu2_2), 17 (relu3_3), 26 (relu4_3)
        # Or just use the first few layers for low-level texture. 
        # Let's use up to relu5_4, causing a deep perceptual match. 
        # But commonly for texture/style: relu1_2, relu2_2, relu3_3, relu4_3
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x]) # relu1_1
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x]) # relu2_1
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x]) # relu3_1
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x]) # relu4_1
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x]) # relu5_1
            
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        # Expect inputs to be normalized similarly to ImageNet if possible, 
        # but standard [-1, 1] input usually works fine if consistently applied.
        # Ideally, we denormalize [-1,1] to [0,1] then normalize with ImageNet mean/std.
        # But simply passing functional inputs often yields good enough gradients.
        
        h_x = x
        h_y = y
        
        h1_x = self.slice1(h_x)
        h1_y = self.slice1(h_y)
        
        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)
        
        h3_x = self.slice3(h2_x)
        h3_y = self.slice3(h2_y)
        
        h4_x = self.slice4(h3_x)
        h4_y = self.slice4(h3_y)
        
        h5_x = self.slice5(h4_x)
        h5_y = self.slice5(h4_y)
        
        loss = torch.mean((h1_x - h1_y) ** 2) + \
               torch.mean((h2_x - h2_y) ** 2) + \
               torch.mean((h3_x - h3_y) ** 2) + \
               torch.mean((h4_x - h4_y) ** 2) + \
               torch.mean((h5_x - h5_y) ** 2)
               
        return loss
