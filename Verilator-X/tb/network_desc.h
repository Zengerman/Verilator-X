#pragma once
#include <vector>
#include <string>

struct LayerDesc {
    std::string type;   // "conv" / "pool" / "residual" / "depthwise"
    int in_h, in_w, in_ch, out_ch, kernel;
    int tiles;
    int weight_bytes;
};

inline std::vector<LayerDesc> VGG16_LAYERS = {
    {"conv", 16,16, 3,64, 3, 4, 1728},
    {"pool", 16,16,64,64, 2, 1, 0},
    {"conv",  8, 8,64,64, 3, 1, 36864},
    {"pool",  8, 8,64,64, 2, 1, 0},
    {"conv",  4, 4,64,64, 3, 1, 36864},
};

inline std::vector<LayerDesc> RESNET50_LAYERS = {
    {"conv", 16,16, 3,64, 3, 4, 1728},
    {"conv", 16,16,64,64, 3, 4, 36864},
    {"residual",16,16,64,64,0,0,0},
    {"pool", 16,16,64,64, 2, 1, 0},
    {"conv",  8, 8,64,64, 3, 1, 36864},
    {"conv",  8, 8,64,64, 3, 1, 36864},
    {"residual",8,8,64,64,0,0,0},
    {"pool",  8, 8,64,64, 2, 1, 0},
    {"conv",  4, 4,64,64, 3, 1, 36864},
};

inline std::vector<LayerDesc> ALEXNET_LAYERS = {
    {"conv", 16,16, 3, 64, 7, 4, 9408},
    {"pool", 16,16,64, 64, 2, 1, 0},
    {"conv",  8, 8,64,128, 5, 1, 307200},
    {"pool",  8, 8,128,128,2, 1, 0},
    {"conv",  4, 4,128,192,3, 1, 442368},
    {"conv",  4, 4,192,192,3, 1, 442368},
    {"conv",  4, 4,192,256,3, 1, 589824},
};

inline std::vector<LayerDesc> MOBILENETV2_LAYERS = {
    {"conv",      16,16, 3, 64, 3, 4, 1728},       // 0 conv1
    {"depthwise", 16,16,64, 64, 3, 4, 576},        // 1 dw1
    {"conv",      16,16,64, 64, 1, 4, 4096},       // 2 pw1
    {"residual",  16,16,64, 64, 0, 0, 0},          // 3 res1
    {"pool",      16,16,64, 64, 2, 1, 0},          // 4 pool1
    {"conv",       8, 8,64,128, 3, 1, 73728},      // 5 conv2
    {"depthwise",  8, 8,128,128,3, 1, 1152},       // 6 dw2
    {"conv",       8, 8,128,128,1, 1, 16384},      // 7 pw2
    {"residual",   8, 8,128,128,0, 0, 0},          // 8 res2
    {"pool",       8, 8,128,128,2, 1, 0},          // 9 pool2
    {"conv",       4, 4,128,256,3, 1, 442368},     // 10 conv3
};

inline std::vector<LayerDesc> GOOGLENET_LAYERS = {
    {"conv",     16,16, 3, 64, 3, 4, 1728},
    {"pool",     16,16,64, 64, 2, 1, 0},
    {"conv",      8, 8,64, 64, 1, 1, 4096},   // b1: 1×1
    {"conv",      8, 8,64, 32, 1, 1, 2048},   // b2a: 1×1
    {"conv",      8, 8,32,128, 3, 1, 36864},  // b2: 3×3
    {"conv",      8, 8,64, 16, 1, 1, 1024},   // b3a: 1×1
    {"conv",      8, 8,16, 32, 5, 1, 12800},  // b3: 5×5
    {"conv",      8, 8,64, 32, 1, 1, 2048},   // b4: 1×1
    // concat 是纯线连接，不是层，所以不在这里
    {"pool",      8, 8,256,256,2, 1, 0},
    {"conv",      4, 4,256,256,3, 1, 589824},
};
