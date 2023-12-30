#pragma once

#include <OpenImageDenoise/oidn.hpp>
#include <vector>
#include <string>

enum OidnMode {
    NONE,
    SIMPLE,
    AUX
};

class OidnDenoiser {
private:
    OidnDenoiser();

public:
    ~OidnDenoiser();
    void denoise(float *color, float *output, int width, int height, int channels);
    void denoise(float *color, float *normal, float *albedo, float *output, int width, int height, int channels);
    void unset_and_set_mode(OidnMode mode);
    void check_error();
    void set_weights(std::string &weight_path);
    void reset_filter();

    static OidnDenoiser *get_instance();

private:
    oidn::DeviceRef m_oidn_device;
    oidn::FilterRef m_filter;
    OidnMode m_mode = OidnMode::NONE;
    std::string m_weight_path{};
    std::vector<char> m_weights{};
};