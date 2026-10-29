"""Four original DGI measurement operators."""

import hashlib

import torch
from torch import nn

from functions.motionblur import Kernel
from functions.svd_operators import CS, Denoising, SuperResolution


class MotionBlurOperator:
    def __init__(self, kernel_size, device):
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(3, 3, kernel_size, bias=False, groups=3),
        ).to(device)
        # The source initializes Blurkernel, then replaces its kernel with another draw.
        _ = Kernel(size=(kernel_size, kernel_size), intensity=0.5).kernelMatrix
        kernel = Kernel(size=(kernel_size, kernel_size), intensity=0.5).kernelMatrix
        with torch.no_grad():
            self.conv[1].weight.copy_(torch.as_tensor(kernel, dtype=torch.float32, device=device))
        self.conv.requires_grad_(False)

    def A(self, image):
        return self.conv(image)

    def At(self, measurement):
        # Retain the identity backprojection used by the original MotionBlurOperator.
        return measurement


def get_degradation(task, img_size, deg_scale, device):
    if task == "sr_avgpool":
        return SuperResolution(3, img_size, int(deg_scale), device)
    if task == "denoising":
        return Denoising(3, img_size, device)
    if task == "cs_blockbased":
        return CS(3, img_size, deg_scale, device)
    if task == "deblur_motion":
        return MotionBlurOperator(int(deg_scale), device)
    raise ValueError(f"Unsupported task: {task}")


def operator_fingerprint(operator):
    """Identify the actual sampled CS basis or blur kernel, not just the seed."""
    digest = hashlib.sha256()
    tensors = {name: value for name, value in vars(operator).items() if torch.is_tensor(value)}
    if hasattr(operator, "conv"):
        tensors.update(operator.conv.state_dict())
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
