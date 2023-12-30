#pragma once

#include <OpenImageDenoise/oidn.hpp>

enum OidnMode {
    NONE, SIMPLE, AUX
};

class OidnDenoiser {
private:
    OidnDenoiser();

public:
    ~OidnDenoiser();
    void denoise(float *color, float *output, int width, int height, int channels);
    void denoise(float *color, float *normal, float *albedo, float *output, int width, int height, int channels);
    void unsetAndSetMode(OidnMode mode);
    void check_error();

    static OidnDenoiser *get_instance();

private:
    oidn::DeviceRef m_oidn_device;
    oidn::FilterRef m_filter;
    OidnMode m_mode = OidnMode::NONE;
};