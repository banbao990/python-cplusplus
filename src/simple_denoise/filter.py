import os
import sys

from OpenGL.GL import *
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
import numpy as np
from enum import Enum

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "../"))
from utils.ogl.gl_helper import OpenGLHelper as glh
from utils.ogl.compute_task import ComputeTask
from utils.ui import UI
from simple_denoise.pynis import NIS


class KernelType(Enum):
    NONE = 0
    AVERAGE = 1
    GAUSSIAN = 2
    MEDIAN = 3
    NIS = 4


class FilterTasks(ComputeTask):

    def __init__(self, ui: UI):
        self.reserved0: float = 0.5
        self.LOCAL_SIZE = 16
        self.name = "Simple Denoise"
        super().__init__(self.name, "denoise/color.comp")

        self.TYPES = [str(i) for i in KernelType.__members__]
        # self.TYPES = [(i[0].upper() + i[1:].lower()) for i in self.TYPES]
        self.need_kernel_size = [False, True, False, True, False]

        self.kernel_type: KernelType = KernelType.NONE
        self.kernel_size = 3
        self.sigma = 1.0
        # NIS
        self.nis_task: NIS = None
        self.output_tex_size_same_with_window = False

        self.use_tonemapping = True

        # TODO: tooooo ugly, should be managed by TextureManager
        self.window_size = [ui.width, ui.height]
        self.texture_size = ui.texture_size
        self.physical_texture_size = ui.texture_size  # real texture size
        self.output_texture = self.create_texture(*ui.texture_size)

    def get_name(self):
        return self.name

    def create_texture(self, width, height):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1)

        # data_test = np.ones((width, height, 4), dtype=np.float32) * 0.5
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, data_test)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        return texture

    def run(self, **kwargs):
        if self.kernel_type != KernelType.NIS:
            self.run_normal(**kwargs)
        elif self.nis_task.is_bilinear():
            # bilinear
            self.run_normal(**kwargs)
        else:
            self.run_nis(**kwargs)

    def run_normal(self, **kwargs):
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
        glBindImageTexture(0, tex_in, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
        glGenerateMipmap(GL_TEXTURE_2D)
        # glUniform1i(glGetUniformLocation(self.program, "img_input"), 0) # we have already bind it in the shader
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glBindImageTexture(1, self.output_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
        # glUniform1i(glGetUniformLocation(self.program, "img_output"), 1) # we have already bind it in the shader
        ktv = self.kernel_type.value
        if (self.nis_task is not None) and (self.nis_task.is_bilinear()):
            ktv = KernelType.NONE.value
        glUniform4i(glGetUniformLocation(self.program, "v1"), self.kernel_size, ktv, self.use_tonemapping, 0)
        glUniform4f(glGetUniformLocation(self.program, "v2"), self.sigma, 0.0, 0.0, 0.0)
        glDispatchCompute(*group_size)
        # make sure writing to image has finished before read
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def run_nis(self, **kwargs):
        program = self.nis_task.get_program()
        glUseProgram(program)

        self.nis_task.update(self.texture_size, self.window_size)

        # attach constant buffer(binding = 0)
        self.nis_task.bind_cbuffer(self.use_tonemapping)

        binding = -1  # texture binding points
        # attach images
        tex = self.output_texture
        binding += 1
        location = glGetUniformLocation(program, "out_texture")
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindImageTexture(binding, tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
        glUniform1i(location, binding)

        tex = kwargs.get('tex_input')
        binding += 1
        location = glGetUniformLocation(program, "SPIRV_Cross_Combinedin_texturesamplerLinearClamp")
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindImageTexture(binding, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
        glUniform1i(location, binding)

        # packed params
        tex = self.nis_task.coef_scale_fp16_texture
        binding += 1
        location = glGetUniformLocation(program, "SPIRV_Cross_Combinedcoef_scalersamplerLinearClamp")
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindImageTexture(binding, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F)
        glUniform1i(location, binding)

        tex = self.nis_task.coef_usm_fp16_texture
        binding += 1
        location = glGetUniformLocation(program, "SPIRV_Cross_Combinedcoef_usmsamplerLinearClamp")
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindImageTexture(binding, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F)
        glUniform1i(location, binding)

        # run
        w, h = self.nis_task.get_optimal_dispatch_size()
        glDispatchCompute(w, h, 1)
        # make sure writing to image has finished before read
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # exit
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def set(self, **kwargs):
        self.use_tonemapping = kwargs.get('use_tonemapping', False)

    def render_ui(self):
        if self.kernel_type != KernelType.NIS or not self.nis_task.is_NV_scaler():
            self.output_tex_size_same_with_window = False
        value_changed = False
        if imgui.tree_node(self.name, imgui.TREE_NODE_DEFAULT_OPEN):
            ktv: int = self.kernel_type.value
            vc, ktv = imgui.combo("kernel type", ktv, self.TYPES)
            self.kernel_type = KernelType(ktv)
            # TODO: maybe need to change texture size(NIS is not same size as input)
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
                imgui.text_ansi("2 sigma: kernel size = {}".format(int(np.ceil(self.sigma * 2))))
                value_changed = value_changed or vc
            elif self.kernel_type == KernelType.NIS:
                if self.nis_task is None:
                    self.nis_task = NIS(self.window_size, self.texture_size)
                self.nis_task.render_ui()
                if (self.nis_task.is_NV_scaler()):
                    self.output_tex_size_same_with_window = True
            self.resize(*self.texture_size)
            imgui.tree_pop()

        return value_changed

    def resize(self, width, height):
        self.texture_size = [width, height]

        if self.kernel_type == KernelType.NIS:
            self.nis_task.resize(width, height)

        # check should be update
        new_size = None
        if (self.output_tex_size_same_with_window):
            new_size = self.window_size
        else:
            new_size = self.texture_size

        if (new_size == self.physical_texture_size):
            return

        self.physical_texture_size = new_size

        if self.output_texture is not None:
            glDeleteTextures(1, [self.output_texture])
        self.output_texture = self.create_texture(*new_size)
