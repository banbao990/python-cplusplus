import setup_oidn_example as oidn
import torch
import drjit as dr
import mitsuba as mi
import os
import imgui

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class OidnDenoiser:
    def __init__(self) -> None:
        oidn.init()

        # weight
        weights_files = os.listdir(os.path.join(CURRENT_DIR, "weights"))
        weights_files_names = [i.split(".")[0].split("_")[-1]
                               for i in weights_files]
        weights_files_names.insert(0, "None")
        # first no weight set
        self.oidn_weight_index = 0
        self.weights_files = weights_files
        self.weights_files_names = weights_files_names

    def denoise_simple(self, img: torch.tensor):
        img_clean = torch.zeros_like(img).to(img.device)
        oidn.denoise(img, img_clean, img.shape[1], img.shape[0], img.shape[2])
        return img_clean

    def denoise_albedo_normal(self, img_color: torch.tensor, img_albedo: torch.tensor, img_normal: torch.tensor):
        img_clean = torch.zeros_like(img_color).to(img_color.device)
        oidn.denoise_with_normal_and_albedo(
            img_color, img_normal, img_albedo, img_clean, img_color.shape[1], img_color.shape[0], img_color.shape[2])
        return img_clean

    def denoise(self, img: mi.TensorXf, use_albedo_and_normal=False):
        ret = None
        if use_albedo_and_normal:
            # dr.sync_device() # wait for mi.render() to finish
            img_color = img[:, :, 0:3].torch()
            img_albedo = img[:, :, 3:6].torch()
            img_normal = img[:, :, 6:9].torch()
            dr.sync_device()  # wait for 3 lines above to finish
            ret = self.denoise_albedo_normal(img_color, img_albedo, img_normal)
        else:
            img = img.torch()
            dr.sync_device()
            ret = self.denoise_simple(img)
        return ret

    def set_weights(self, weights_path: str):
        oidn.set_weights(weights_path)

    def render_ui(self, use_albedo_and_normal) -> (bool, bool):
        value_changed = False

        # weight
        vc, self.oidn_weight_index = imgui.combo(
            "Weights", self.oidn_weight_index, self.weights_files_names)
        if vc:
            weight_path = ""
            if self.oidn_weight_index != 0:
                weight_path = os.path.join(
                    CURRENT_DIR, "weights", self.weights_files[self.oidn_weight_index - 1])
            self.set_weights(weight_path)
        value_changed = value_changed or vc

        # aux
        if self.oidn_weight_index == 0:
            vc, use_albedo_and_normal = imgui.checkbox(
                "Use Albedo and Normal", use_albedo_and_normal)
            value_changed = value_changed or vc
        else:
            # weight file indicates whether to use albedo and normal
            tmp = use_albedo_and_normal
            use_albedo_and_normal = self.weights_files[self.oidn_weight_index - 1].find(
                "alb_nrm") != -1
            vc = tmp != use_albedo_and_normal
            value_changed = value_changed or vc

        return value_changed, use_albedo_and_normal
