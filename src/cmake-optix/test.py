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
from datetime import datetime

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    scene_file = os.path.join(
        CURRENT_DIR, "../../assets/ignore/scenes/veach-ajar/scene.xml")
    if (not os.path.exists(scene_file)):
        print(
            "\033[91mScene File Not Found, Please Run 'python prepare.py' in the root dir\033[0m")
        exit(-1)

    scene: mi.Scene = mi.load_file(scene_file)
    width, height = scene.sensors()[0].film().size()

    ui = UI(width, height)

    optix_denoiser_on = False
    denoiser = None
    index = 0
    update_frame = True

    use_same_seed = False
    same_seed = 0
    img = None

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
        vc, use_same_seed = imgui.checkbox("Use Same Seed", use_same_seed)
        seed = index
        if (vc):
            same_seed = seed
        if (use_same_seed):
            seed = same_seed
        value_changed = value_changed or vc

        if (value_changed or update_frame):
            img = mi.render(scene=scene, spp=4, seed=seed)

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

    ui.close()

    # add these lines to avoid jit_shutdown() error
    # we should use the img variable to notifiy the jit compiler
    result_dir = os.path.join(CURRENT_DIR, "../../results")
    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir)
    timestamp = datetime.today().strftime('%Y-%m-%d-%H%M%S')
    mi.util.write_bitmap(os.path.join(result_dir, "output-{}.png".format(timestamp)), img)