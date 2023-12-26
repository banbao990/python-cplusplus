import cmake_optix_example as optix
import imgui
import mitsuba as mi
import torch


class OptixDenoiser:
    def __init__(self):
        self.module = optix
        self.aux = False
        self.temporal = False

    def denoise(self, noisy: torch.Tensor):
        img = self.module.denoise(noisy, self.aux, self.temporal)
        return img

    def render_ui(self, integrator):
        value_changed = False
        if imgui.tree_node("Denoise Options", imgui.TREE_NODE_DEFAULT_OPEN):
            vc, self.aux = imgui.checkbox("Use Albedo and Normal", self.aux)
            value_changed = value_changed or vc

            vc, self.temporal = imgui.checkbox("Use Temporal", self.temporal)
            value_changed = value_changed or vc

            imgui.tree_pop()

        aovs = "albedo:albedo,sh_normal:sh_normal"

        if (self.aux):
            integrator = mi.load_dict({
                'type': 'aov',
                'aovs': aovs,
                'integrator': integrator
            })

        return value_changed, integrator
