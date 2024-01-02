import torch
import torchvision
import numpy as np
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv


def read_exr(path: str) -> torch.Tensor:
    """
    Read exr image and convert to torch tensor.
    """

    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    img = torch.from_numpy(img)
    img = img.to(torch.float32)
    img = img.to("cuda")
    return img


def read_png(path: str) -> torch.Tensor:
    """
    Read png image and convert to torch tensor.
    """
    img = cv.imread(path, cv.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    img = torch.from_numpy(img)
    img = img.to("cuda")
    return img


def gen_noise(size: tuple, samples: int, blur: bool = False) -> torch.Tensor:
    """
    Generate noise with given size, elements in noise are in range [0, 1)
    """
    desized_size = size[0] * size[1] * 3
    desired_shape = (size[0], size[1], 3)

    # total gray images
    gray = torch.ones(desired_shape, dtype=torch.float32, device="cuda")
    gray *= 0.5

    samples = min(samples, desized_size)

    # generate noise
    noise = torch.rand(samples, dtype=torch.float32, device="cuda") * 2.0 - 1.0

    # extend and shuffle
    noise = torch.cat([noise, torch.zeros(
        desized_size - samples, dtype=torch.float32, device="cuda")])
    noise = noise[torch.randperm(noise.size(0))]
    noise = noise.reshape(desired_shape)

    if blur:
        # blur the noise with 3x3 kernel, noise is (size[0], size[1], 3), kernel is (3, 3)
        # first change nouce to (3, size[0], size[1]), then blur, then change back to (size[0], size[1], 3)
        noise = noise.permute(2, 0, 1)
        blurred_noise = torchvision.transforms.GaussianBlur(
            kernel_size=3, sigma=3)(noise)
        noise = blurred_noise.permute(1, 2, 0)

    return gray + noise * 2.0

# site: https://github.com/cuteday/KiRaRay/blob/main/src/render/passes/tonemapping/tonemapping.cu


def tonemap_aces(img: torch.Tensor) -> torch.Tensor:
    """
    Tonemap image with ACES.
    """
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    img *= 0.6
    img = torch.clamp((img * (A * img + B)) / (img * (C * img + D) + E), 0, 1)
    img = torch.pow(img, 0.45454545)  # gamma 2.2
    return img
