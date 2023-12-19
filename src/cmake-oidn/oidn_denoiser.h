#pragma once

#include <OpenImageDenoise/oidn.hpp>

class OidnDenoiser {
private:
    OidnDenoiser();

public:
    ~OidnDenoiser();
    void denoise(float *color, float *output, int width, int height, int channels);

    static OidnDenoiser *get_instance();
private:
    oidn::DeviceRef m_oidn_device;
};