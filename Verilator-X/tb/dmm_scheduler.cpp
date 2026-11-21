#include "dmm_scheduler.h"
#include <iostream>

DataflowTemplate dmm_select(const LayerDesc& layer, int pe_rows, int pe_cols,
                            int buffer_kb, bool use_dmm) {
    if (!use_dmm) {
        if (layer.type == "pool")      return DataflowTemplate::POOL_STREAM;
        if (layer.type == "residual")  return DataflowTemplate::ELEMENTWISE;
        if (layer.type == "depthwise") return DataflowTemplate::LINE;
        return DataflowTemplate::LINE;
    }

    // ---- 规则引擎：conv / depthwise 用精确公式 ----
    if (layer.type == "conv" || layer.type == "depthwise") {
        long wbytes   = (long)layer.out_ch * layer.kernel * layer.kernel * layer.in_ch;
        if (layer.type == "depthwise") {
            wbytes = (long)layer.in_ch * layer.kernel * layer.kernel;  // DW 每通道独立
        }
        long bufbytes = (long)buffer_kb * 1024;
        if (wbytes > bufbytes) return DataflowTemplate::LINE;
        return DataflowTemplate::BLOCK;
    }

    // ---- 非 conv 层：决策树 ----
    DMMFeatures f;
    f.op_type   = (layer.type == "pool") ? 1.0f : 2.0f;
    f.in_h = layer.in_h; f.in_w = layer.in_w;
    f.in_ch = layer.in_ch; f.out_ch = layer.out_ch;
    f.kernel = layer.kernel;
    f.pe_rows = pe_rows; f.pe_cols = pe_cols;
    f.buffer_kb = buffer_kb;
    f.tiles = layer.tiles;
    return dmm_tree_infer_impl(f);
}

int dmm_reload_count(DataflowTemplate t, int tiles) {
    if (t == DataflowTemplate::POOL_STREAM) return 0;
    if (t == DataflowTemplate::ELEMENTWISE) return 0;
    if (t == DataflowTemplate::LINE)        return (tiles > 0 ? tiles * 4 : 1);
    return 1;
}

void dmm_print(const std::string& name, DataflowTemplate t) {
    std::cout << "[DMM] " << name << " -> ";
    switch (t) {
        case DataflowTemplate::LINE:        std::cout << "LINE"; break;
        case DataflowTemplate::BLOCK:       std::cout << "BLOCK"; break;
        case DataflowTemplate::POOL_STREAM: std::cout << "POOL_STREAM"; break;
        case DataflowTemplate::ELEMENTWISE: std::cout << "ELEMENTWISE"; break;
    }
    std::cout << std::endl;
}
