import cmake_optix_example as optix
import imgui
import mitsuba as mi


class OptixDenoiser:
    def __init__(self):
        self.module = optix
        self.aux = False

    def denoise(self, noisy):
        img = None
        if (self.aux):
            color = noisy[..., :3].torch()
            albedo = noisy[..., 3:6].torch()
            normal = noisy[..., 6:9].torch()
            img = self.module.denoise_aux(color, albedo, normal)
        else:
            img = self.module.denoise(noisy.torch())
        return img

    def render_ui(self, integrator):
        value_changed = False
        if imgui.tree_node("Denoise Options", imgui.TREE_NODE_DEFAULT_OPEN):
            vc, self.aux = imgui.checkbox("Use Albedo and Normal", self.aux)
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
