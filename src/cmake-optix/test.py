import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CURRENT_DIR, "../"))
sys.path.append(CURRENT_DIR)

from utils.ui import UI
import mitsuba as mi
from utils.images import tonemap_aces
import imgui
from optix import OptixDenoiser
import time

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    scene_file = os.path.join(CURRENT_DIR, "../../assets/ignore/scenes/veach-ajar/scene.xml")
    if (not os.path.exists(scene_file)):
        print("\033[91mScene File Not Found, Please Run 'python prepare.py' in the root dir\033[0m")
        exit(-1)

    scene: mi.Scene = mi.load_file(scene_file)
    width, height = scene.sensors()[0].film().size()

    ui = UI(width, height)

    optix_denoiser_on = False
    denoiser = None
    index = 0
    update_frame = True

    while not ui.should_close():
        if (not update_frame):
            time.sleep(1 / 60)
        ui.begin_frame()
        value_changed = False
        vc, update_frame = imgui.checkbox("Update Frame", update_frame)
        value_changed = value_changed or vc
        vc, optix_denoiser_on = imgui.checkbox(
            "Optix Denoiser On", optix_denoiser_on)
        value_changed = value_changed or vc

        if (value_changed or update_frame):
            img = mi.render(scene=scene, spp=4, seed=int(index * 1000))

            if (optix_denoiser_on):
                if (denoiser == None):
                    denoiser = OptixDenoiser()
                img = denoiser.denoise(img.torch())
            else:
                img = img.torch()

            img = tonemap_aces(img)

            # ui.write_texture_cpu(img.cpu().numpy())
            ui.write_texture_gpu(img)

        ui.end_frame()
        index += 1
    if (denoiser is not None):
        denoiser.free()
    ui.close()
