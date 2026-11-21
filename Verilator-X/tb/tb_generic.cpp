// ============================================================
// Unified testbench for 5 networks:
//   VGG16, ResNet50, AlexNet, MobileNetV2, GoogLeNet
//
// Build with one of:
//   -DNETWORK_VGG16
//   -DNETWORK_RESNET50
//   -DNETWORK_ALEXNET
//   -DNETWORK_MOBILENETV2
//   -DNETWORK_GOOGLENET
//
// Runtime mode via env:
//   VERILATOR_X_MODE=baseline | pts | dmm | pts_dmm
// ============================================================
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <verilated.h>

#include "network_desc.h"
#include "dmm_scheduler.h"
#include "pts_scorer.h"
#include "tl_layers.h"

// -------- DUT type & network layers --------
#if defined(NETWORK_VGG16)
#include "Vvgg16_accelerator.h"
using DUT = Vvgg16_accelerator;
static const char* NET_NAME = "VGG16";
static std::vector<LayerDesc> NET_LAYERS = VGG16_LAYERS;
#elif defined(NETWORK_RESNET50)
#include "Vresnet50_accelerator.h"
using DUT = Vresnet50_accelerator;
static const char* NET_NAME = "ResNet50";
static std::vector<LayerDesc> NET_LAYERS = RESNET50_LAYERS;
#elif defined(NETWORK_ALEXNET)
#include "Valexnet_accelerator.h"
using DUT = Valexnet_accelerator;
static const char* NET_NAME = "AlexNet";
static std::vector<LayerDesc> NET_LAYERS = ALEXNET_LAYERS;
#elif defined(NETWORK_MOBILENETV2)
#include "Vmobilenetv2_accelerator.h"
using DUT = Vmobilenetv2_accelerator;
static const char* NET_NAME = "MobileNetV2";
static std::vector<LayerDesc> NET_LAYERS = MOBILENETV2_LAYERS;
#elif defined(NETWORK_GOOGLENET)
#include "Vgooglenet_accelerator.h"
using DUT = Vgooglenet_accelerator;
static const char* NET_NAME = "GoogLeNet";
static std::vector<LayerDesc> NET_LAYERS = GOOGLENET_LAYERS;
#else
#error "No NETWORK_xxx defined"
#endif

// -------- Global constants --------
constexpr int IN_SIZE    = 16;
constexpr int IN_CH      = 3;
constexpr int FINAL_SIZE = 4;
constexpr int MAX_CYCLES = 20000000;
constexpr int PE_ROWS    = 64;
constexpr int PE_COLS    = 64;
constexpr int BUF_KB     = 64;

static vluint64_t sim_time = 0;

static inline void clock_cycle(DUT* d) {
    d->clk = 0; d->eval(); ++sim_time;
    d->clk = 1; d->eval(); ++sim_time;
}

// ============================================================
// PTS context: single (pool) or dual (residual) input
// ============================================================
struct PtsCtx {
    std::vector<uint8_t> a;   // main input (pool) / branch a (residual)
    std::vector<uint8_t> b;   // skip branch (residual only, empty for pool)
    int h = 0, w = 0, ch = 0;
    bool is_residual = false;
};

// ============================================================
// DbgIf: abstract interface each network implements
// ============================================================
struct DbgIf {
    virtual ~DbgIf() {}
    virtual PtsCtx read_pts(int li) = 0;
    virtual void   write_pts(int li, const std::vector<uint8_t>& y) = 0;
    virtual bool   can_skip(int li) = 0;
};

// ============================================================
// VGG16 DbgIf
//   layers: 0 conv1, 1 pool1, 2 conv2, 3 pool2, 4 conv3
//   pool1 in = dbg_conv1_out, out = dbg_pool1_out (8×8×64)
//   pool2 in = dbg_conv2_out, out = dbg_pool2_out (4×4×64)
// ============================================================
#if defined(NETWORK_VGG16)
struct DbgVGG : DbgIf {
    DUT* d; explicit DbgVGG(DUT* x) : d(x) {}
    PtsCtx read_pts(int li) override {
        PtsCtx c;
        if (li == 1) {
            c.h = 16; c.w = 16; c.ch = 64;
            c.a.resize(16*16*64);
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) for (int k=0;k<64;k++)
                c.a[((i*16+j)*64)+k] = d->dbg_conv1_out[i][j][k];
        } else {
            c.h = 8; c.w = 8; c.ch = 64;
            c.a.resize(8*8*64);
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<64;k++)
                c.a[((i*8+j)*64)+k] = d->dbg_conv2_out[i][j][k];
        }
        return c;
    }
    void write_pts(int li, const std::vector<uint8_t>& y) override {
        if (li == 1) {
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<64;k++)
                d->dbg_pool1_out[i][j][k] = y[((i*8+j)*64)+k];
        } else {
            for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<64;k++)
                d->dbg_pool2_out[i][j][k] = y[((i*4+j)*64)+k];
        }
    }
    bool can_skip(int li) override { return (li == 1 || li == 3); }
};
#endif

// ============================================================
// ResNet50 DbgIf
//   layers: 0 conv1, 1 conv2, 2 res1, 3 pool1, 4 conv3,
//           5 conv4, 6 res2, 7 pool2, 8 conv5
//   res1 = conv2 + conv1, out = dbg_res1_out (16×16×64)
//   pool1 in = dbg_res1_out, out = dbg_pool1_out (8×8×64)
//   res2 = conv4 + conv3, out = dbg_res2_out (8×8×64)
//   pool2 in = dbg_res2_out, out = dbg_pool2_out (4×4×64)
// ============================================================
#if defined(NETWORK_RESNET50)
struct DbgRes : DbgIf {
    DUT* d; explicit DbgRes(DUT* x) : d(x) {}
    PtsCtx read_pts(int li) override {
        PtsCtx c;
        if (li == 2) {                       // res1
            c.h = 16; c.w = 16; c.ch = 64; c.is_residual = true;
            c.a.resize(16*16*64); c.b.resize(16*16*64);
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) for (int k=0;k<64;k++) {
                c.a[((i*16+j)*64)+k] = d->dbg_conv2_out[i][j][k];
                c.b[((i*16+j)*64)+k] = d->dbg_conv1_out[i][j][k];
            }
        } else if (li == 3) {                // pool1
            c.h = 16; c.w = 16; c.ch = 64;
            c.a.resize(16*16*64);
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) for (int k=0;k<64;k++)
                c.a[((i*16+j)*64)+k] = d->dbg_res1_out[i][j][k];
        } else if (li == 6) {                // res2
            c.h = 8; c.w = 8; c.ch = 64; c.is_residual = true;
            c.a.resize(8*8*64); c.b.resize(8*8*64);
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<64;k++) {
                c.a[((i*8+j)*64)+k] = d->dbg_conv4_out[i][j][k];
                c.b[((i*8+j)*64)+k] = d->dbg_conv3_out[i][j][k];
            }
        } else {                              // pool2 (li=7)
            c.h = 8; c.w = 8; c.ch = 64;
            c.a.resize(8*8*64);
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<64;k++)
                c.a[((i*8+j)*64)+k] = d->dbg_res2_out[i][j][k];
        }
        return c;
    }
    void write_pts(int li, const std::vector<uint8_t>& y) override {
        if (li == 2) {
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) for (int k=0;k<64;k++)
                d->dbg_res1_out[i][j][k] = y[((i*16+j)*64)+k];
        } else if (li == 3) {
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<64;k++)
                d->dbg_pool1_out[i][j][k] = y[((i*8+j)*64)+k];
        } else if (li == 6) {
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<64;k++)
                d->dbg_res2_out[i][j][k] = y[((i*8+j)*64)+k];
        } else {
            for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<64;k++)
                d->dbg_pool2_out[i][j][k] = y[((i*4+j)*64)+k];
        }
    }
    bool can_skip(int li) override { return (li==2 || li==3 || li==6 || li==7); }
};
#endif

// ============================================================
// AlexNet DbgIf
//   layers: 0 conv1, 1 pool1, 2 conv2, 3 pool2, 4 conv3, 5 conv4, 6 conv5
// ============================================================
#if defined(NETWORK_ALEXNET)
struct DbgAlex : DbgIf {
    DUT* d; explicit DbgAlex(DUT* x) : d(x) {}
    PtsCtx read_pts(int li) override {
        PtsCtx c;
        if (li == 1) {
            c.h = 16; c.w = 16; c.ch = 64;
            c.a.resize(16*16*64);
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) for (int k=0;k<64;k++)
                c.a[((i*16+j)*64)+k] = d->dbg_conv1_out[i][j][k];
        } else {
            c.h = 8; c.w = 8; c.ch = 128;
            c.a.resize(8*8*128);
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<128;k++)
                c.a[((i*8+j)*128)+k] = d->dbg_conv2_out[i][j][k];
        }
        return c;
    }
    void write_pts(int li, const std::vector<uint8_t>& y) override {
        if (li == 1) {
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<64;k++)
                d->dbg_pool1_out[i][j][k] = y[((i*8+j)*64)+k];
        } else {
            for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<128;k++)
                d->dbg_pool2_out[i][j][k] = y[((i*4+j)*128)+k];
        }
    }
    bool can_skip(int li) override { return (li == 1 || li == 3); }
};
#endif

// ============================================================
// MobileNetV2 DbgIf
//   layers: 0 conv1, 1 dw1, 2 pw1, 3 res1, 4 pool1,
//           5 conv2, 6 dw2, 7 pw2, 8 res2, 9 pool2, 10 conv3
//   res1 = pw1 + conv1, out = dbg_res1_out (16×16×64)
//   pool1 in = dbg_res1_out, out = dbg_pool1_out (8×8×64)
//   res2 = pw2 + conv2, out = dbg_res2_out (8×8×128)
//   pool2 in = dbg_res2_out, out = dbg_pool2_out (4×4×128)
// ============================================================
#if defined(NETWORK_MOBILENETV2)
struct DbgMBN : DbgIf {
    DUT* d; explicit DbgMBN(DUT* x) : d(x) {}
    PtsCtx read_pts(int li) override {
        PtsCtx c;
        if (li == 3) {                       // res1
            c.h = 16; c.w = 16; c.ch = 64; c.is_residual = true;
            c.a.resize(16*16*64); c.b.resize(16*16*64);
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) for (int k=0;k<64;k++) {
                c.a[((i*16+j)*64)+k] = d->dbg_pw1_out[i][j][k];
                c.b[((i*16+j)*64)+k] = d->dbg_conv1_out[i][j][k];
            }
        } else if (li == 4) {                // pool1
            c.h = 16; c.w = 16; c.ch = 64;
            c.a.resize(16*16*64);
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) for (int k=0;k<64;k++)
                c.a[((i*16+j)*64)+k] = d->dbg_res1_out[i][j][k];
        } else if (li == 8) {                // res2
            c.h = 8; c.w = 8; c.ch = 128; c.is_residual = true;
            c.a.resize(8*8*128); c.b.resize(8*8*128);
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<128;k++) {
                c.a[((i*8+j)*128)+k] = d->dbg_pw2_out[i][j][k];
                c.b[((i*8+j)*128)+k] = d->dbg_conv2_out[i][j][k];
            }
        } else {                              // pool2 (li=9)
            c.h = 8; c.w = 8; c.ch = 128;
            c.a.resize(8*8*128);
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<128;k++)
                c.a[((i*8+j)*128)+k] = d->dbg_res2_out[i][j][k];
        }
        return c;
    }
    void write_pts(int li, const std::vector<uint8_t>& y) override {
        if (li == 3) {
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) for (int k=0;k<64;k++)
                d->dbg_res1_out[i][j][k] = y[((i*16+j)*64)+k];
        } else if (li == 4) {
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<64;k++)
                d->dbg_pool1_out[i][j][k] = y[((i*8+j)*64)+k];
        } else if (li == 8) {
            for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<128;k++)
                d->dbg_res2_out[i][j][k] = y[((i*8+j)*128)+k];
        } else {
            for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<128;k++)
                d->dbg_pool2_out[i][j][k] = y[((i*4+j)*128)+k];
        }
    }
    bool can_skip(int li) override { return (li==3 || li==4 || li==8 || li==9); }
};
#endif

// ============================================================
// GoogLeNet DbgIf
//   layers: 0 conv1, 1 pool1, 2..7 Inception branches, 8 pool2, 9 conv_f
//   NOTE: 为简化，GoogLeNet 只对 pool1 做 PTS（pool2 需要 concat 输出，
//         RTL 侧没有 dbg_pool2_out 端口；如需支持，在 RTL 里补
//         input dbg_pool2_out[0:3][0:3][0:255] 和 output dbg_concat_in[0:7][0:7][0:255]）
// ============================================================
#if defined(NETWORK_GOOGLENET)
struct DbgGoogle : DbgIf {
    DUT* d; explicit DbgGoogle(DUT* x) : d(x) {}
    PtsCtx read_pts(int li) override {
        PtsCtx c;
        // 仅 pool1
        c.h = 16; c.w = 16; c.ch = 64;
        c.a.resize(16*16*64);
        for (int i=0;i<16;i++) for (int j=0;j<16;j++) for (int k=0;k<64;k++)
            c.a[((i*16+j)*64)+k] = d->dbg_conv1_out[i][j][k];
        return c;
    }
    void write_pts(int li, const std::vector<uint8_t>& y) override {
        for (int i=0;i<8;i++) for (int j=0;j<8;j++) for (int k=0;k<64;k++)
            d->dbg_pool1_out[i][j][k] = y[((i*8+j)*64)+k];
    }
    bool can_skip(int li) override { return (li == 1); }   // 只 pool1
};
#endif

// ============================================================
// Factory
// ============================================================
static DbgIf* make_dbg(DUT* d) {
#if   defined(NETWORK_VGG16)
    return new DbgVGG(d);
#elif defined(NETWORK_RESNET50)
    return new DbgRes(d);
#elif defined(NETWORK_ALEXNET)
    return new DbgAlex(d);
#elif defined(NETWORK_MOBILENETV2)
    return new DbgMBN(d);
#elif defined(NETWORK_GOOGLENET)
    return new DbgGoogle(d);
#endif
}

// ============================================================
// Weight loading
// ============================================================
static int load_weights(DUT* dut, const std::vector<std::vector<uint8_t>>& W,
                        const std::vector<int>& reload) {
    int cycles = 0;
    int conv_id = 0;
    for (size_t i = 0; i < NET_LAYERS.size(); ++i) {
        const auto& t = NET_LAYERS[i].type;
        if (t == "pool" || t == "residual") continue;
        if (W[i].empty()) continue;
        for (int r = 0; r < reload[i]; ++r) {
            dut->weight_we = 1;
            for (size_t k = 0; k < W[i].size(); ++k) {
                dut->weight_addr = ((uint32_t)conv_id << 20) | (uint32_t)(k & 0xFFFF);
                dut->weight_data = W[i][k];
                clock_cycle(dut); ++cycles;
            }
            dut->weight_we = 0;
        }
        ++conv_id;
    }
    return cycles;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const char* mode_env = std::getenv("VERILATOR_X_MODE");
    const std::string mode = mode_env ? mode_env : "baseline";
    const bool use_dmm = (mode == "dmm" || mode == "pts_dmm");
    const bool use_pts = (mode == "pts" || mode == "pts_dmm");

    std::cout << "=== " << NET_NAME
              << " | Mode=" << mode
              << " | DMM=" << (use_dmm?"ON":"OFF")
              << " | PTS=" << (use_pts?"ON":"OFF")
              << " ===" << std::endl;

    // ---- DMM decision ----
    std::vector<DataflowTemplate> tmpl(NET_LAYERS.size());
    std::vector<int>               reload(NET_LAYERS.size(), 0);
    for (size_t i = 0; i < NET_LAYERS.size(); ++i) {
        tmpl[i]   = dmm_select(NET_LAYERS[i], PE_ROWS, PE_COLS, BUF_KB, use_dmm);
        reload[i] = dmm_reload_count(tmpl[i], NET_LAYERS[i].tiles);
        char name[64];
        std::snprintf(name, sizeof(name), "%zu_%s", i, NET_LAYERS[i].type.c_str());
        dmm_print(name, tmpl[i]);
    }

    // ---- PTS scoring ----
    DUT* dut = new DUT();
    DbgIf* dbg = make_dbg(dut);

    std::vector<bool> skip(NET_LAYERS.size(), false);
    for (size_t i = 0; i < NET_LAYERS.size(); ++i) {
        const std::string& t = NET_LAYERS[i].type;
        if (!use_pts) continue;
        if (!dbg->can_skip((int)i)) continue;
        if (t == "pool") {
            PtsFeatures f{1.0, 0.8, 0.1, 0.5, 0.9};
            skip[i] = pts_is_non_critical(pts_score(f), 0.6);
        } else if (t == "residual") {
            PtsFeatures f{0.9, 0.7, 0.2, 0.4, 0.9};
            skip[i] = pts_is_non_critical(pts_score(f), 0.6);
        }
        if (skip[i]) {
            std::cout << "[PTS] skip layer " << i << " (" << t << ")" << std::endl;
        }
    }

    // ---- init DUT ----
    auto t0 = std::chrono::high_resolution_clock::now();

    dut->reset_n   = 0;
    dut->start     = 0;
    dut->pts_mode  = use_pts ? 1 : 0;
    dut->pts_ready = 0;
    dut->weight_we = 0;
    for (int i = 0; i < 10; ++i) clock_cycle(dut);
    dut->reset_n = 1;

    // input
    for (int i = 0; i < IN_SIZE; ++i)
        for (int j = 0; j < IN_SIZE; ++j)
            for (int c = 0; c < IN_CH; ++c)
                dut->feature_in[i][j][c] = (uint8_t)((i*IN_SIZE + j + c*17) % 256);

    // random weights
    std::vector<std::vector<uint8_t>> W(NET_LAYERS.size());
    for (size_t i = 0; i < NET_LAYERS.size(); ++i) {
        W[i].resize(NET_LAYERS[i].weight_bytes);
        for (size_t k = 0; k < W[i].size(); ++k)
            W[i][k] = (uint8_t)((k*7 + 3) % 251 + 1);
    }

    // ---- weight loading ----
    auto tw0 = std::chrono::high_resolution_clock::now();
    int weight_cycles = load_weights(dut, W, reload);
    auto tw1 = std::chrono::high_resolution_clock::now();

    // ---- main loop ----
    auto tc0 = std::chrono::high_resolution_clock::now();

    dut->start = 1; clock_cycle(dut);
    dut->start = 0;
    int compute_cycles = 1;

    while (!dut->done && compute_cycles < MAX_CYCLES) {
        if (dut->pts_mode && dut->wait_pts) {
            int li = (int)dut->pts_layer_idx;
            if (li >= 0 && li < (int)NET_LAYERS.size() && skip[li]) {
                PtsCtx c = dbg->read_pts(li);
                std::vector<uint8_t> y;
                if (c.is_residual) {
                    y.resize(c.a.size());
                    tl_residual_add(c.a.data(), c.b.data(), y.data(), c.h, c.ch);
                } else {
                    int oh = c.h / 2, ow = c.w / 2;
                    y.resize((size_t)oh*ow*c.ch);
                    tl_max_pool_2x2(c.a.data(), y.data(), c.h, c.ch);
                }
                dbg->write_pts(li, y);
            }
            dut->pts_ready = 1;
            clock_cycle(dut); ++compute_cycles;
            dut->pts_ready = 0;
        } else {
            clock_cycle(dut); ++compute_cycles;
        }
    }

    auto tc1 = std::chrono::high_resolution_clock::now();
    auto t1  = std::chrono::high_resolution_clock::now();

    // ---- count non-zero outputs ----
    int out_h = NET_LAYERS.back().in_h;
    int out_w = NET_LAYERS.back().in_w;
    int out_c = NET_LAYERS.back().out_ch;
    int total = out_h * out_w * out_c;
    int nonzero = 0;
    for (int i = 0; i < out_h; ++i)
        for (int j = 0; j < out_w; ++j)
            for (int c = 0; c < out_c; ++c)
                if (dut->feature_out[i][j][c]) ++nonzero;

    auto total_ns   = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto weight_ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(tw1 - tw0).count();
    auto compute_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tc1 - tc0).count();

    // ---- report ----
    std::cout << "\n--- Report ---\n";
    std::cout << "Network: "          << NET_NAME           << std::endl;
    std::cout << "Mode: "             << mode               << std::endl;
    std::cout << "DMM enabled: "      << (use_dmm?"YES":"NO") << std::endl;
    std::cout << "PTS enabled: "      << (use_pts?"YES":"NO") << std::endl;
    std::cout << "Weight cycles: "    << weight_cycles      << std::endl;
    std::cout << "Compute cycles: "   << compute_cycles     << std::endl;
    std::cout << "Total cycles: "     << (weight_cycles + compute_cycles) << std::endl;
    std::cout << "Non-zero outputs: " << nonzero << " / " << total << std::endl;
    std::cout << "Weight time: "      << weight_ns  / 1e6  << " ms" << std::endl;
    std::cout << "Compute time: "     << compute_ns / 1e6  << " ms" << std::endl;
    std::cout << "Total time: "       << total_ns   / 1e6  << " ms" << std::endl;
    std::cout << "Done signal: "      << (dut->done?"YES":"NO") << std::endl;

    delete dbg;
    delete dut;
    return 0;
}
