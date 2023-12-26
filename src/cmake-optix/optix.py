import cmake_optix_example as optix


class OptixDenoiser:
    def __init__(self):
        self.module = optix

    def denoise(self, input):
        return self.module.denoise(input)

    def free(self):
        '''
        must call this function to free the memory
        '''
        self.module.free()
