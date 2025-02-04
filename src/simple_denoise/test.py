from OpenGL.GL import *
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui

import argparse
import os
import sys
import time
import mitsuba as mi
from datetime import datetime
import torch
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "../"))
from utils.images import tonemap_aces, read_exr, read_png
from utils.ogl.gl_helper import OpenGLHelper as glh
from simple_denoise.filter import FilterTasks
from utils.ui import UI


def save_img(img, num_acc: int = 0):
    result_dir = os.path.join(CURRENT_DIR, "../../results")
    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir)
    timestamp = datetime.today().strftime('%Y-%m-%d-%H%M%S')
    mi.util.write_bitmap(os.path.join(result_dir, "output-{}-{}.png".format(timestamp, num_acc)), img)


def test_ui(args: argparse.Namespace):
    src_img = read_exr(os.path.join(CURRENT_DIR, "../../assets/images/100spp.exr"))
    # BGR to RGB
    src_img = src_img[::, ::, [2, 1, 0]].clone()
    src_img_depth = torch.ones_like(src_img) * 0.5  # no depth
    height, width, _ = src_img.shape

    ui = UI(width, height, args.gpu)

    denoise_on = False
    denoise_task = None
    use_tonemapping = True

    scale = 1.0
    size_ori = (width, height)
    size_render = size_ori

    while not ui.should_close():
        time.sleep(1 / 60)
        ui.begin_frame()
        vc, use_tonemapping = imgui.checkbox("Use Tonemap", use_tonemapping)
        vc, denoise_on = imgui.checkbox("Denoise On", denoise_on)
        if (vc):
            if denoise_on:
                if denoise_task is None:
                    denoise_task: FilterTasks = FilterTasks(ui)
                ui.set_compute_task(denoise_task, False)
            else:
                ui.set_compute_task(None, False)
        if (ui.compute_task != None):
            ui.compute_task.set(use_tonemapping=use_tonemapping)
            _ = ui.compute_task.render_ui()

        vc1 = imgui.button("Set 2x")
        if vc1:
            scale = 0.5
        vc2, scale = imgui.slider_float("Scale", scale, 0.1, 1.0)
        if vc1 or vc2:
            size_render = [int(scale * i) for i in size_ori]
            ui.check_and_update_texture_size(*size_render)

        imgui.text("Original Size: {} x {}".format(*size_ori))
        imgui.text("Render Size: {} x {}".format(*size_render))

        need_depth = False
        if (denoise_on):
            need_depth = denoise_task.need_depth()
        if (need_depth):
            img_depth = src_img_depth.clone()
            denoise_task.record_depth(img_depth, False)

        start_x = (width - size_render[0]) // 2
        start_y = (height - size_render[1]) // 2
        end_x = start_x + size_render[0]
        end_y = start_y + size_render[1]
        img = src_img[start_y:end_y, start_x:end_x, ::].clone()

        if (use_tonemapping and not denoise_on):
            img = tonemap_aces(img)

        ui.write_texture_gpu(img)

        ui.end_frame()

    if denoise_task is not None:
        denoise_task.release()
        denoise_task = None
        ui.set_compute_task(None, False)

    ui.close()


def test_render(args: argparse.Namespace):
    mi.set_variant("cuda_ad_rgb")

    scene: mi.Scene = None
    if args.cbox:
        cbox = mi.cornell_box()
        cbox["sensor"]["film"]["width"] = 1024
        cbox["sensor"]["film"]["height"] = 1024
        scene = mi.load_dict(cbox)
    else:
        scene_file = os.path.join(CURRENT_DIR, "../../assets/ignore/scenes/veach-bidir/scene.xml")
        if (not os.path.exists(scene_file)):
            print("\033[91mScene File Not Found, Please Run 'python prepare.py' in the root dir\033[0m")
            exit(-1)
        scene = mi.load_file(scene_file)

    width, height = scene.sensors()[0].film().size()

    ui = UI(width, height, args.gpu)

    use_same_seed = False
    same_seed = 0
    index = 0
    update_frame = True
    spp = 1

    img = None
    acc = False
    img_acc = None
    num_acc = 0
    max_acc = 0
    stop_render_when_max_acc = True
    use_tonemapping = True
    denoise_on = False
    denoise_task = None
    depth_integrator = mi.load_dict({'type': 'depth'})

    scale = 1.0
    scene_params = mi.traverse(scene)
    film_size_key: str = None
    for k in scene_params.keys():
        if k.endswith("film.size"):
            film_size_key = k
            break
    size_ori = (width, height)
    size_render = size_ori

    while not ui.should_close():
        if (not update_frame):
            time.sleep(1 / 60)
        ui.begin_frame()
        value_changed = False
        _, update_frame = imgui.checkbox("Update Frame", update_frame)
        vc, spp = imgui.slider_int("spp", spp, 1, 16)
        value_changed = value_changed or vc
        vc, acc = imgui.checkbox("Accumulate", acc)
        if (vc):
            num_acc = 0
            img_acc = None
        value_changed = value_changed or vc
        if (acc):
            vc, stop_render_when_max_acc = imgui.checkbox("Stop Render When Max Acc", stop_render_when_max_acc)
            value_changed = value_changed or vc
            vc, max_acc = imgui.slider_int("Max Accumulate", max_acc, 0, 500)
            if (vc):
                # force update frame, as we stop update frame when max_acc is reached
                update_frame = True
                num_acc = 0
                img_acc = None
            value_changed = value_changed or vc
            imgui.text_ansi("Accumulate Frames: {}".format(num_acc + 1))

        integrator = scene.integrator()
        vc, use_same_seed = imgui.checkbox("Use Same Seed", use_same_seed)
        value_changed = value_changed or vc
        seed = index + int(time.time())
        if (vc):
            same_seed = seed
        if (use_same_seed):
            seed = same_seed
        vc, use_tonemapping = imgui.checkbox("Use Tonemap", use_tonemapping)
        value_changed = value_changed or vc
        save = imgui.button("Save")

        vc, denoise_on = imgui.checkbox("Denoise On", denoise_on)
        if (vc):
            if denoise_on:
                if denoise_task is None:
                    denoise_task: FilterTasks = FilterTasks(ui)
                ui.set_compute_task(denoise_task, False)
            else:
                ui.set_compute_task(None, False)
        if (ui.compute_task != None):
            ui.compute_task.set(use_tonemapping=use_tonemapping)
            _ = ui.compute_task.render_ui()
        value_changed = value_changed or vc

        vc1 = imgui.button("Set 2x")
        if vc1:
            scale = 0.5

        vc2, scale = imgui.slider_float("Scale", scale, 0.1, 1.0)

        if vc1 or vc2:
            size_render = [int(scale * i) for i in size_ori]
            scene_params[film_size_key] = size_render
            scene_params.update()
            ui.check_and_update_texture_size(*size_render)
            num_acc = 0
            img_acc = None
            value_changed = True

        imgui.text("Original Size: {} x {}".format(*size_ori))
        imgui.text("Render Size: {} x {}".format(*size_render))

        update_frame = update_frame or value_changed
        if (update_frame):
            need_depth = False
            if (denoise_on):
                need_depth = denoise_task.need_depth()
            if (need_depth):
                img_depth = mi.render(scene=scene, spp=1, seed=seed, integrator=depth_integrator)
                denoise_task.record_depth(img_depth[::, ::, 0].torch())

            img = mi.render(scene=scene, spp=spp, seed=seed, integrator=integrator)

            img = img.torch()

            if (acc):
                if (img_acc == None):
                    img_acc = img.clone()
                    num_acc = 1
                elif (max_acc == 0 or (num_acc + 1 < max_acc)):
                    img_acc = img_acc + img
                    num_acc += 1
                else:
                    # TODO: bug?
                    # save compute resource
                    update_frame = False or not stop_render_when_max_acc
                img = img_acc / num_acc

            if save:
                save_img(img, num_acc)

            if (use_tonemapping and not denoise_on):
                img = tonemap_aces(img)

            ui.write_texture_gpu(img)

        ui.end_frame()
        index += 1

    if denoise_task is not None:
        denoise_task.release()
        denoise_task = None
        ui.set_compute_task(None, False)

    ui.close()

    # add these lines to avoid jit_shutdown() error
    # we should use the img variable to notifiy the jit compiler
    save_img(img, num_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Write texture on GPU")
    parser.add_argument("--render", action="store_true", help="Render Mode")
    parser.add_argument("--cbox", action="store_true", help="Use Cornell Box")
    args = parser.parse_args()

    parser.print_help()
    if (args.render):
        test_render(args)
    else:
        test_ui(args)
