import torch
import torchvision.transforms as T

def log_spectrum(x, normalize=True):

    gray = T.functional.rgb_to_grayscale(x)
    fft = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
    mag = torch.log1p(fft.abs())
    if normalize:
        mean, std = torch.std_mean(mag, dim=(-2, -1), keepdim=True)
        mag = (mag - mean) / (std + 1e-6)

    return mag

