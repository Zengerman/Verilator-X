#pragma once
#include <string>
#include "network_desc.h"

struct DMMFeatures {
    float op_type;   // 0=conv, 1=pool, 2=residual
    float in_h, in_w, in_ch, out_ch, kernel;
    float pe_rows, pe_cols, buffer_kb;
    float tiles;
};

enum class DataflowTemplate { LINE, BLOCK, POOL_STREAM, ELEMENTWISE };

DataflowTemplate dmm_tree_infer_impl(const DMMFeatures& f);   // 由决策树导出
DataflowTemplate dmm_select(const LayerDesc& layer, int pe_rows, int pe_cols,
                            int buffer_kb, bool use_dmm);
int  dmm_reload_count(DataflowTemplate t, int tiles);
void dmm_print(const std::string& name, DataflowTemplate t);
