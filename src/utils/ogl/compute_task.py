from OpenGL.GL import *
import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(CURRENT_DIR)
from gl_helper import OpenGLHelper as glh
import imgui


class ComputeTask(object):
    def __init__(self, name, shader_path):
        self.name = name
        self.program = glh.create_compute_program(shader_path)

    def run(self, **kwargs):
        raise NotImplementedError

    def render_ui(self):
        raise NotImplementedError

    def set(self, **kwargs):
        raise NotImplementedError

    def release(self):
        glDeleteProgram(self.program)


class ComputeTaskTest(ComputeTask):
    def __init__(self):
        super().__init__("Test", "samples/highlight.comp")
        self.center = [0.5, 0.5]
        self.radius = 0.1

    def run(self, **kwargs):
        group_size = kwargs.get('group_size')
        assert group_size is not None
        group_size = list(group_size)
        len_group_size = len(group_size)
        assert len_group_size == 2

        group_size[0] = np.ceil(group_size[0] / 16)
        group_size[1] = np.ceil(group_size[1] / 16)
        group_size.append(1)
        group_size = [int(x) for x in group_size]

        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, kwargs.get('tex_input'))
        glUniform1i(glGetUniformLocation(self.program, "img_output"), 0)
        glUniform4f(glGetUniformLocation(self.program, "highlight_control"), self.center[0], self.center[1], self.radius, 0.0)
        glDispatchCompute(*group_size)
        # make sure writing to image has finished before read
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def render_ui(self):
        value_changed = False
        if imgui.tree_node("Compute Shader Test", imgui.TREE_NODE_DEFAULT_OPEN):
            # slide float for radius
            vc, self.radius = imgui.slider_float("radius", self.radius, 0.0, 1.0)
            value_changed = value_changed or vc
            vc, self.center[0] = imgui.slider_float("center x", self.center[0], 0.0, 1.0)
            value_changed = value_changed or vc
            vc, self.center[1] = imgui.slider_float("center y", self.center[1], 0.0, 1.0)
            value_changed = value_changed or vc

            imgui.tree_pop()

        return value_changed
