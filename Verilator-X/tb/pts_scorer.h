// .h
#pragma once
struct PtsFeatures {
    double op_type, data_range, gradient_sensitivity,
           structural_depth, output_utilization;
};
double pts_score(const PtsFeatures& f);
bool   pts_is_non_critical(double score, double theta = 0.6);
