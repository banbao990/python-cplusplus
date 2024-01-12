import os
import sys

from OpenGL.GL import *
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "../"))
from utils.ogl.gl_helper import OpenGLHelper as glh
from utils.ogl.compute_task import ComputeTask
from utils.ui import UI
from enum import Enum


class KernelType(Enum):
    NONE = 0
    AVERAGE = 1
    GAUSSIAN = 2
    MEDIAN = 3


class Filters(ComputeTask):

    def __init__(self, ui: UI):
        self.LOCAL_SIZE = 16
        self.name = "Simple Denoise"
        super().__init__(self.name, "denoise/color.comp")

        # first char of the word is big, others are small
        self.TYPES = [(str(i)[0].upper() + str(i)[1:].lower()) for i in KernelType.__members__]
        self.need_kernel_size = [False, True, False, True]

        self.kernel_type: KernelType = KernelType.MEDIAN
        self.kernel_size = 3
        self.sigma = 1.0

        self.use_tonemapping = True

        # TODO: tooooo ugly, should be managed by TextureManager
        self.output_texture = self.create_texture(*ui.texture_size)

    def get_name(self):
        return self.name

    def create_texture(self, width, height):
        texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # data_test = np.ones((width, height, 4), dtype=np.float32) * 0.5
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, data_test)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, ctypes.c_void_p(0))
        glBindTexture(GL_TEXTURE_2D, 0)

        return texture

    def run(self, **kwargs):
        group_size = kwargs.get('group_size')
        assert group_size is not None
        group_size = list(group_size)
        assert len(group_size) == 2

        group_size[0] = np.ceil(group_size[0] / self.LOCAL_SIZE)
        group_size[1] = np.ceil(group_size[1] / self.LOCAL_SIZE)
        group_size.append(1)
        group_size = [int(x) for x in group_size]

        glUseProgram(self.program)
        tex_in = kwargs.get('tex_input')
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_in)
        glBindImageTexture(0, tex_in, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
        glUniform1i(glGetUniformLocation(self.program, "img_input"), 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glBindImageTexture(1, self.output_texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
        glUniform1i(glGetUniformLocation(self.program, "img_output"), 1)
        glUniform4i(glGetUniformLocation(self.program, "v1"), self.kernel_size, self.kernel_type.value, self.use_tonemapping, 0)
        glUniform4f(glGetUniformLocation(self.program, "v2"), self.sigma, 0.0, 0.0, 0.0)
        glDispatchCompute(*group_size)
        # make sure writing to image has finished before read
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def set(self, **kwargs):
        self.use_tonemapping = kwargs.get('use_tonemapping', False)

    def render_ui(self):
        value_changed = False
        if imgui.tree_node(self.name, imgui.TREE_NODE_DEFAULT_OPEN):
            ktv: int = self.kernel_type.value
            vc, ktv = imgui.combo("kernel type", ktv, self.TYPES)
            self.kernel_type = KernelType(ktv)
            value_changed = value_changed or vc

            if self.need_kernel_size[ktv]:
                vc, self.kernel_size = imgui.slider_int("kernel size", self.kernel_size, 1, 5)
                ks = self.kernel_size * 2 - 1
                imgui.text_ansi("kernel size: {} x {}".format(ks, ks))
                value_changed = value_changed or vc
                if self.kernel_type == KernelType.MEDIAN:
                    imgui.text_ansi("Median Filter is Slow! Bubble Sort! Max Kernel Size = 9")

            if self.kernel_type == KernelType.GAUSSIAN:
                vc, self.sigma = imgui.slider_float("sigma", self.sigma, 0.1, 5.0)
                imgui.text_ansi("2sigma: kernel size = {}".format(int(np.ceil(self.sigma * 2))))
                value_changed = value_changed or vc

            imgui.tree_pop()

        return value_changed

    def resize(self, width, height):
        if self.output_texture is not None:
            glDeleteTextures(1, [self.output_texture])
        self.output_texture = self.create_texture(width, height)
