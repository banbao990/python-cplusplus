#include "oidn_denoiser.h"
#include <iostream>

OidnDenoiser *OidnDenoiser::get_instance() {
    static OidnDenoiser s_instance;
    return &s_instance;
}

OidnDenoiser::OidnDenoiser() {
    m_oidn_device = oidn::newDevice(oidn::DeviceType::CUDA);
    m_oidn_device.commit();

    m_filter = m_oidn_device.newFilter("RT");
}

OidnDenoiser::~OidnDenoiser() {
    m_filter.release();
    m_oidn_device.release();
    std::cout << "OidnDenoiser released successfully" << std::endl;
}

void OidnDenoiser::denoise(float *color, float *output, int width, int height, int channels) {
    unsetAndSetMode(OidnMode::SIMPLE);

    oidn::Format format = oidn::Format(int(oidn::Format::Float) + channels - 1);

    m_filter.setImage("color", color, format, width, height);
    m_filter.setImage("output", output, format, width, height);
    m_filter.set("hdr", true);
    m_filter.commit();
    m_filter.execute();

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
    unsetAndSetMode(OidnMode::AUX);

    oidn::Format format = oidn::Format(int(oidn::Format::Float) + channels - 1);

    m_filter.setImage("color", color, format, width, height);
    m_filter.setImage("normal", normal, format, width, height);
    m_filter.setImage("albedo", albedo, format, width, height);
    m_filter.setImage("output", output, format, width, height);
    m_filter.set("hdr", true);
    m_filter.commit();
    m_filter.execute();

    check_error();
}

void OidnDenoiser::unsetAndSetMode(OidnMode mode) {
    if(m_mode != mode) {
        m_filter.unsetImage("color");
        m_filter.unsetImage("normal");
        m_filter.unsetImage("albedo");
        m_filter.unsetImage("output");
        m_mode = mode;
    }

}
