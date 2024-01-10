# site: https://github.com/woAIxuexiSR/dynamic-neural-radiosity/blob/denoise/utils/ui.py

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
from imgui.integrations.glfw import GlfwRenderer

from cuda import cudart
import argparse
import numpy as np
import os
import sys
import time
import mitsuba as mi
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "../"))
from utils.my_queues import FixedSizeQueue
from utils.images import tonemap_aces

VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoords;

out vec2 texCoords;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    texCoords = aTexCoords;
}
"""

FRAGMENT_SHADER = """
#version 330 core

out vec4 FragColor;

in vec2 texCoords;
uniform sampler2D Image;

void main() {
    FragColor = texture(Image, texCoords);
}
"""


def check_cuda_error(cres: cudart.cudaError_t):
    if cres != cudart.cudaError_t.cudaSuccess:
        print(str(cres))
        print("\033[31mCUDA error: {}\033[0m".format(cres))


class UI:

    def __init__(self, width, height, gpu: bool = False, name="Simple UI"):
        print("[UI] Write texture on GPU: {}".format(gpu))
        self.gpu = gpu
        self.width = width
        self.height = height
        self.name = name
        self.fps = FixedSizeQueue(30)

        # initialize glfw
        if not glfw.init():
            print("Failed to initialize GLFW")
            exit()

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(width, height, name, None, None)

        if not window:
            print("Failed to create window")
            glfw.terminate()
            exit()

        glfw.make_context_current(window)

        glViewport(0, 0, width, height)

        glfw.swap_interval(0)
        self.window = window

        # initialize imgui
        imgui.create_context()
        self.impl = GlfwRenderer(window)

        # create shader, vao, texture
        self.program = self.create_program()
        self.vao = self.create_vao()

        self.texture_size = (-1, -1)
        self.texture = None
        self.pbo = None
        self.bufobj = None
        self.check_and_update_texture_size(width, height)

    def close(self):
        if self.gpu:
            cres, = cudart.cudaGraphicsUnregisterResource(self.bufobj)
            check_cuda_error(cres)
            glDeleteBuffers(1, [self.pbo])

        glDeleteTextures(1, [self.texture])
        glDeleteVertexArrays(1, [self.vao])
        glDeleteProgram(self.program)

        self.impl.shutdown()
        glfw.destroy_window(self.window)
        glfw.terminate()

    def create_program(self):
        vertex = OpenGL.GL.shaders.compileShader(
            VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment = OpenGL.GL.shaders.compileShader(
            FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

        return OpenGL.GL.shaders.compileProgram(vertex, fragment)

    def create_vao(self):

        # flip vertically
        quad = np.array([
            # position 2, texcoord 2
            -1.0, 1.0, 0.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 1.0,

            -1.0, 1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0
        ], dtype=np.float32)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 *
                              quad.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 *
                              quad.itemsize, ctypes.c_void_p(2 * quad.itemsize))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glDeleteBuffers(1, [vbo])

        return vao

    def create_pbo(self, w, h):

        data = np.zeros((w * h * 3), dtype=np.float32)
        pbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, pbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return pbo

    def create_texture(self):

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, self.texture_size[0], self.texture_size[1], 0, GL_RGB, GL_FLOAT, None)

        return texture

    def should_close(self):
        return glfw.window_should_close(self.window)

    def set_should_close(self, value):
        glfw.set_window_should_close(self.window, value)

    def begin_frame(self):
        self.fps.push(glfw.get_time())

        imgui.new_frame()
        imgui.begin("Test")
        time_average = (self.fps.back() - self.fps.front()) / self.fps.size()
        if time_average > 0:
            imgui.text("FPS: {:.2f}".format(1.0 / time_average))

    # img: (height, width, 3) np.float32
    def write_texture_cpu(self, img):
        self.check_and_update_texture_size(img.shape[1], img.shape[0])

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.texture_size[0], self.texture_size[1], GL_RGB, GL_FLOAT, img)

    # img: (height, width, 3) torch.float32
    def write_texture_gpu(self, img):
        if not self.gpu:
            self.write_texture_cpu(img.cpu().numpy())
            return

        self.check_and_update_texture_size(img.shape[1], img.shape[0])
        cres, = cudart.cudaGraphicsMapResources(1, self.bufobj, 0)
        check_cuda_error(cres)
        cres, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(
            self.bufobj)
        check_cuda_error(cres)
        cres, = cudart.cudaMemcpy(
            ptr, img.data_ptr(), size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        check_cuda_error(cres)
        cres, = cudart.cudaGraphicsUnmapResources(1, self.bufobj, 0)
        check_cuda_error(cres)

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, int(self.pbo))
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.texture_size[0], self.texture_size[1], GL_RGB, GL_FLOAT, ctypes.c_void_p(0))

    def end_frame(self):
        imgui.end()

        imgui.render()
        imgui.end_frame()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        self.impl.render(imgui.get_draw_data())
        self.impl.process_inputs()
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def check_and_update_texture_size(self, width, height):
        if width == self.texture_size[0] and height == self.texture_size[1]:
            return
        if self.gpu:
            if self.bufobj != None:
                cres, = cudart.cudaGraphicsUnregisterResource(self.bufobj)
                check_cuda_error(cres)
            if self.pbo != None:
                glDeleteBuffers(1, [self.pbo])

        if self.texture != None:
            glDeleteTextures(1, [self.texture])

        self.texture_size = (width, height)
        print(self.texture_size)
        self.texture = self.create_texture()

        if self.gpu:
            self.pbo = self.create_pbo(self.texture_size[0], self.texture_size[1])
            cres, self.bufobj = cudart.cudaGraphicsGLRegisterBuffer(int(self.pbo), cudart.cudaGraphicsRegisterFlags(0))
            check_cuda_error(cres)

##########################################################################################


def test_ui(args: argparse.Namespace):
    mi.set_variant("cuda_ad_rgb")
    bmp: mi.Bitmap = mi.Bitmap(os.path.join(CURRENT_DIR, "../../assets/images/100spp.exr"))
    img = mi.TensorXf(bmp).torch().to("cuda")[::, ::, 0:3].clone()
    img = tonemap_aces(img)

    width, height = 1280, 720

    ui = UI(width, height, args.gpu)

    while not ui.should_close():
        time.sleep(1 / 60)
        ui.begin_frame()
        ui.write_texture_gpu(img)
        ui.end_frame()

    ui.close()


def test_render(args: argparse.Namespace):
    mi.set_variant("cuda_ad_rgb")

    scene_file = os.path.join(CURRENT_DIR, "../../assets/ignore/scenes/veach-bidir/scene.xml")
    if (not os.path.exists(scene_file)):
        print("\033[91mScene File Not Found, Please Run 'python prepare.py' in the root dir\033[0m")
        exit(-1)

    scene: mi.Scene = mi.load_file(scene_file)
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
    use_tonemapping = True
    scale = 1.0
    set2x = False
    scene_params = mi.traverse(scene)
    size_ori = (width, height)
    size_render = size_ori

    while not ui.should_close():
        if (not update_frame):
            time.sleep(1 / 60)
        ui.begin_frame()
        value_changed = False
        vc, update_frame = imgui.checkbox("Update Frame", update_frame)
        value_changed = value_changed or vc
        vc, spp = imgui.slider_int("spp", spp, 1, 16)
        value_changed = value_changed or vc
        vc, acc = imgui.checkbox("Accumulate", acc)
        if (vc):
            num_acc = 0
            img_acc = None
        imgui.text_ansi("Accumulate Frames: {}".format(num_acc + 1))
        value_changed = value_changed or vc

        integrator = scene.integrator()
        vc, use_same_seed = imgui.checkbox("Use Same Seed", use_same_seed)
        value_changed = value_changed or vc
        seed = index
        if (vc):
            same_seed = seed
        if (use_same_seed):
            seed = same_seed
        vc, use_tonemapping = imgui.checkbox("Use Tonemap", use_tonemapping)
        value_changed = value_changed or vc

        vc2x, set2x = imgui.checkbox("Set 2x", set2x)
        if set2x:
            scale = 0.5
        value_changed = value_changed or vc2x

        vc, scale = imgui.slider_float("Scale", scale, 0.1, 1.0)

        if vc or vc2x:
            size_render = [int(scale * i) for i in size_ori]
            scene_params["PerspectiveCamera.film.size"] = size_render
            scene_params.update()
        value_changed = value_changed or vc

        imgui.text("Original Size: {} x {}".format(*size_ori))
        imgui.text("Render Size: {} x {}".format(*size_render))

        if (value_changed or update_frame):
            img = mi.render(scene=scene, spp=spp, seed=seed,
                            integrator=integrator)

            img = img.torch()

            if (acc):
                if (img_acc == None):
                    img_acc = img[::, ::, 0:3:1]
                else:
                    img_acc = img_acc + img[::, ::, 0:3:1]
                num_acc += 1
                img[::, ::, 0:3:1] = img_acc / num_acc

            if (use_tonemapping):
                img = tonemap_aces(img)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Write texture on GPU")
    parser.add_argument("--render", action="store_true", help="Renderer Mode")
    args = parser.parse_args()

    parser.print_help()

    if args.render:
        test_render(args)
    else:
        test_ui(args)
