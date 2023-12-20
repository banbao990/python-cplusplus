#include "oidn_denoiser.h"
#include <iostream>

OidnDenoiser *OidnDenoiser::get_instance() {
    static OidnDenoiser s_instance;
    return &s_instance;
}

OidnDenoiser::OidnDenoiser() {
    m_oidn_device = oidn::newDevice(oidn::DeviceType::CUDA);
    m_oidn_device.commit();
}

OidnDenoiser::~OidnDenoiser() {
    m_oidn_device.release();
    std::cout << "OidnDenoiser released successfully" << std::endl;
}

void OidnDenoiser::denoise(float *color, float *output, int width, int height, int channels) {
    oidn::Format format = oidn::Format(int(oidn::Format::Float) + channels - 1);

    oidn::FilterRef filter = m_oidn_device.newFilter("RT");
    filter.setImage("color", color, format, width, height);
    filter.setImage("output", output, format, width, height);
    filter.set("hdr", true);
    filter.commit();
    filter.execute();
    filter.release();

    check_error();
}

void OidnDenoiser::check_error() {
    // check errors
    const char *error_message;
    if (m_oidn_device.getError(error_message) != oidn::Error::None) {
        std::cerr << "[Error] " << error_message << std::endl;
    }
}

void OidnDenoiser::denoise(float *color, float *normal, float *albedo, float *output, int width, int height, int channels) {
    oidn::Format format = oidn::Format(int(oidn::Format::Float) + channels - 1);

    oidn::FilterRef filter = m_oidn_device.newFilter("RT");
    filter.setImage("color", color, format, width, height);
    filter.setImage("normal", normal, format, width, height);
    filter.setImage("albedo", albedo, format, width, height);
    filter.setImage("output", output, format, width, height);
    filter.set("hdr", true);
    filter.commit();
    filter.execute();
    filter.release();

    check_error();
}