# model.py
# multi-modal detector: spatial resnet18 + 5-conv freq branch on dct

import torch
import torch.nn as nn
import torchvision.models as models
import torch_dct

IMG_SIZE = 256

class Detector(nn.Module):
    def __init__(self, mode="fusion"):
        super().__init__()
        self.mode = mode
        
        # spatial branch: pretrained resnet18, fine-tuned (transfer learning piece)
        if mode in ["spatial", "fusion"]:
            weights = models.ResNet18_Weights.DEFAULT
            self.spatial_net = models.resnet18(weights=weights)
            self.spatial_net.fc = nn.Identity()
            # without this the spatial branch's bigger numbers were probably just drowning out freq
            self.spatial_norm = nn.LayerNorm(512)
            # adding this so the spatial branch gets direct gradient signal even in fusion mode
            self.spatial_head = nn.Linear(512, 1)
            
            # imagenet stats for normalization
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # freq branch: grayscale -> 2D DCT -> log1p -> normalize -> 5-conv-block CNN from scratch
        # using dct instead of fft, avoids that weird cross artifact fft spectrums always have
        # no pretrained model does dct input so this one's starting from zero
        if mode in ["freq", "fusion"]:
            self.freq_net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            # LayerNorm on freq branch too
            self.freq_norm = nn.LayerNorm(128)
            # adding this so the freq branch doesn't just get ignored, trying it out
            self.freq_head = nn.Linear(128, 1)
            
        if mode == "fusion":
            # fusion head combines both normalized feature vectors
            self.fusion_head = nn.Sequential(
                nn.Linear(512 + 128, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )

    def extract_spatial(self, x):
        # normalize with imagenet mean/std
        x_norm = (x - self.mean) / self.std
        feat = self.spatial_net(x_norm)
        return self.spatial_norm(feat)

    def extract_freq(self, x):
        # grayscale conversion
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        # 2D DCT (low freq is top-left, no fftshift needed)
        dct = torch_dct.dct_2d(gray)
        mag = torch.log1p(torch.abs(dct))
        # normalize per image
        m = mag.mean(dim=(-2, -1), keepdim=True)
        s = mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        mag_norm = (mag - m) / s
        feat = self.freq_net(mag_norm)
        return self.freq_norm(feat)

    def forward(self, x):
        spatial_out, freq_out = None, None
        
        if self.mode == "spatial":
            s_feat = self.extract_spatial(x)
            main_out = self.spatial_head(s_feat)
            return main_out, main_out, None
            
        elif self.mode == "freq":
            f_feat = self.extract_freq(x)
            main_out = self.freq_head(f_feat)
            return main_out, None, main_out
            
        elif self.mode == "fusion":
            s_feat = self.extract_spatial(x)
            f_feat = self.extract_freq(x)
            
            # side outputs for direct gradient signal even in fusion mode
            spatial_out = self.spatial_head(s_feat)
            freq_out = self.freq_head(f_feat)
            
            # forcing it to sometimes work without spatial, see if freq branch actually learns something
            s_in, f_in = s_feat, f_feat
            if self.training:
                r = torch.rand(1).item()
                if r < 0.25:
                    s_in = torch.zeros_like(s_feat)
                elif r < 0.50:
                    f_in = torch.zeros_like(f_feat)
                    
            fused = torch.cat([s_in, f_in], dim=1)
            main_out = self.fusion_head(fused)
            return main_out, spatial_out, freq_out
