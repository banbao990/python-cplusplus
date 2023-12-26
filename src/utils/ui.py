# site: https://github.com/woAIxuexiSR/dynamic-neural-radiosity/blob/denoise/utils/ui.py

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
from imgui.integrations.glfw import GlfwRenderer

from cuda import cudart

import numpy as np
import os
import sys
import mitsuba as mi

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

    def __init__(self, width, height, name="Simple UI"):
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
        self.texture = self.create_texture()

        self.pbo = self.create_pbo(width, height)
        cres, self.bufobj = cudart.cudaGraphicsGLRegisterBuffer(
            int(self.pbo), cudart.cudaGraphicsRegisterFlags(0))
        check_cuda_error(cres)

    def close(self):

        cres, = cudart.cudaGraphicsUnregisterResource(self.bufobj)
        check_cuda_error(cres)

        glDeleteProgram(self.program)
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.pbo])
        glDeleteTextures(1, [self.texture])

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

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, self.width,
                     self.height, 0, GL_RGB, GL_FLOAT, None)

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

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width,
                        self.height, GL_RGB, GL_FLOAT, img)

    # img: (height, width, 3) torch.float32
    def write_texture_gpu(self, img):
        cres, = cudart.cudaGraphicsMapResources(1, self.bufobj, 0)
        check_cuda_error(cres)
        cres, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(self.bufobj)
        check_cuda_error(cres)
        cres, = cudart.cudaMemcpy(ptr, img.data_ptr(), size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        check_cuda_error(cres)
        cres, = cudart.cudaGraphicsUnmapResources(1, self.bufobj, 0)
        check_cuda_error(cres)

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, int(self.pbo))
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width,
                        self.height, GL_RGB, GL_FLOAT, ctypes.c_void_p(0))

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


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    bmp: mi.Bitmap = mi.Bitmap(os.path.join(
        CURRENT_DIR, "../../assets/images/100spp.exr"))
    img = mi.TensorXf(bmp).torch().to("cuda")[::, ::, 0:3].clone()
    img = tonemap_aces(img)

    width, height = 1280, 720
    ui = UI(width, height)

    while not ui.should_close():
        ui.begin_frame()
        ui.write_texture_gpu(img)
        ui.end_frame()

    ui.close()
