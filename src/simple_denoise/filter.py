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


class Filters(ComputeTask):
    def __init__(self, ui: UI):
        self.LOCAL_SIZE = 16
        self.name = "Simple Denoise"
        super().__init__(self.name, "denoise/color.comp")

        self.TYPES = ["None", "Average"]
        self.kernel_type = 0
        self.kernel_size = 1
        self.use_tonemapping = True

        # TODO: tooooo ugly, should be managed by TextureManager
        # self.output_texture = self.create_texture(*ui.texture_size)

    def get_name(self):
        return self.name

    def create_texture(self, width, height):
        texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)

        return texture

    def run(self, **kwargs):
        group_size = kwargs.get('group_size')
        assert group_size is not None
        group_size = list(group_size)
        len_group_size = len(group_size)
        assert len_group_size == 2

        group_size[0] = np.ceil(group_size[0] / self.LOCAL_SIZE)
        group_size[1] = np.ceil(group_size[1] / self.LOCAL_SIZE)
        group_size.append(1)
        group_size = [int(x) for x in group_size]

        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, kwargs.get('tex_input'))
        # glActiveTexture(GL_TEXTURE1)
        # glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glUniform4i(glGetUniformLocation(self.program, "v1"), self.kernel_size, self.kernel_type, self.use_tonemapping, 0)
        glUniform1i(glGetUniformLocation(self.program, "img_input"), 0)
        # glUniform1i(glGetUniformLocation(self.program, "img_output"), 2)
        glDispatchCompute(*group_size)
        # make sure writing to image has finished before read
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def set(self, **kwargs):
        self.use_tonemapping = kwargs.get('use_tonemapping', False)

    def render_ui(self):
        value_changed = False
        if imgui.tree_node(self.name, imgui.TREE_NODE_DEFAULT_OPEN):
            vc, self.kernel_type = imgui.combo("kernel type", self.kernel_type, self.TYPES)
            value_changed = value_changed or vc

            vc, self.kernel_size = imgui.slider_int("kernel size", self.kernel_size, 1, 5)
            ks = self.kernel_size * 2 - 1
            imgui.text_ansi("kernel size: {} x {}".format(ks, ks))
            value_changed = value_changed or vc

            imgui.tree_pop()

        return value_changed
