import oidn_example as oidn
import torch
import drjit as dr
import mitsuba as mi

class OidDenoiser:
    def __init__(self) -> None:
        oidn.init()

    def denoise_simple(self, img:torch.tensor):
        img_clean = torch.zeros_like(img).to(img.device)
        oidn.denoise(img, img_clean, img.shape[1], img.shape[0], img.shape[2])
        return img_clean
    
    def denoise_albedo_normal(self, img_color:torch.tensor, img_albedo:torch.tensor, img_normal:torch.tensor):
        img_clean = torch.zeros_like(img_color).to(img_color.device)
        oidn.denoise_with_normal_and_albedo(img_color, img_normal, img_albedo, img_clean, img_color.shape[1], img_color.shape[0], img_color.shape[2])
        return img_clean

    def denoise(self, img: mi.TensorXf, use_albedo_and_normal=False):
        ret = None
        if use_albedo_and_normal:
            # dr.sync_device() # wait for mi.render() to finish
            img_color = img[:, :, 0:3].torch()
            img_albedo = img[:, :, 3:6].torch()
            img_normal = img[:, :, 6:9].torch()
            dr.sync_device() # wait for 3 lines above to finish
            ret = self.denoise_albedo_normal(img_color, img_albedo, img_normal)
        else:
            img = img.torch()
            dr.sync_device()
            ret = self.denoise_simple(img)
        return ret