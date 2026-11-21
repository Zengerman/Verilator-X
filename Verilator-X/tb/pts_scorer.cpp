// .cpp
#include "pts_scorer.h"
double pts_score(const PtsFeatures& f) {
    return 0.35*f.op_type + 0.25*f.data_range + 0.15*f.gradient_sensitivity
         + 0.15*f.structural_depth + 0.10*f.output_utilization;
}
bool pts_is_non_critical(double s, double th) { return s >= th; }
