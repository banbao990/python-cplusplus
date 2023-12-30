#include "oidn_denoiser.h"
#include "utils.h"
#include <iostream>
#include <filesystem>

OidnDenoiser *OidnDenoiser::get_instance() {
    static OidnDenoiser s_instance;
    return &s_instance;
}

OidnDenoiser::OidnDenoiser() {
    m_oidn_device = oidn::newDevice(oidn::DeviceType::CUDA);
    m_oidn_device.commit();
    reset_filter();
}

void OidnDenoiser::reset_filter() {
    if (m_filter.getHandle() != nullptr) {
        m_filter.release();
    }
    m_filter = m_oidn_device.newFilter("RT");
}

OidnDenoiser::~OidnDenoiser() {
    if (m_filter.getHandle() != nullptr) {
        m_filter.release();
    }
    if (m_oidn_device.getHandle() != nullptr) {
        m_oidn_device.release();
    }
    std::cout << MI_INFO << "OidnDenoiser released successfully" << std::endl;
}

void OidnDenoiser::denoise(float *color, float *output, int width, int height, int channels) {
    unset_and_set_mode(OidnMode::SIMPLE);

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
        std::cerr << MI_ERROR << error_message << std::endl;
    }
}

void OidnDenoiser::set_weights(std::string &weight_path) {
    if (weight_path.empty()) {
        std::cout << MI_INFO << "Reset weights" << std::endl;
        unset_and_set_mode(OidnMode::NONE);
        reset_filter();
        return;
    }
    std::string filename = std::filesystem::path(weight_path).filename().u8string();
    if (weight_path.compare(m_weight_path) == 0) {
        std::cout << MI_INFO << "Weights already loaded" << std::endl;
    } else {
        std::cout << MI_INFO << "Loading weights from " << filename << std::endl
                  << "    " << weight_path;
        m_weights = load_file(weight_path);
        m_weight_path = weight_path;
    }
    OidnMode mode = filename.find("alb_nrm") ? OidnMode::AUX : OidnMode::SIMPLE;
    unset_and_set_mode(mode);
    m_filter.setData("weights", m_weights.data(), m_weights.size());
    m_filter.commit();
    check_error();
}

void OidnDenoiser::denoise(float *color, float *normal, float *albedo, float *output, int width, int height, int channels) {
    unset_and_set_mode(OidnMode::AUX);

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

void OidnDenoiser::unset_and_set_mode(OidnMode mode) {
    if (m_mode != mode) {
        m_filter.unsetImage("color");
        m_filter.unsetImage("normal");
        m_filter.unsetImage("albedo");
        m_filter.unsetImage("output");
        m_mode = mode;
    }
}
