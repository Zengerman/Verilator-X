#!/bin/bash
set -e
JOBS=${JOBS:-4}
ROOT=$(pwd)

echo "========== Step 1: 训练 DMM 决策树 =========="
if [ ! -f tb/dmm_tree_generated.cpp ] || [ scripts/dmm_train.py -nt tb/dmm_tree_generated.cpp ]; then
    python3 scripts/dmm_train.py
else
    echo "dmm_tree_generated.cpp 已存在，跳过训练"
fi

declare -A NET_RTL NET_TOP NET_DEF NET_LIBS

NET_RTL[VGG16]="rtl/vgg16_accelerator.v rtl/conv_pe_array.v rtl/pool_layer.v"
NET_TOP[VGG16]="vgg16_accelerator"
NET_DEF[VGG16]="NETWORK_VGG16"
NET_LIBS[VGG16]=""

NET_RTL[ResNet50]="rtl/resnet50_accelerator.v rtl/conv_pe_array.v rtl/pool_layer.v rtl/residual_add.v"
NET_TOP[ResNet50]="resnet50_accelerator"
NET_DEF[ResNet50]="NETWORK_RESNET50"
NET_LIBS[ResNet50]=""

NET_RTL[AlexNet]="rtl/alexnet_accelerator.v rtl/conv_pe_array.v rtl/pool_layer.v"
NET_TOP[AlexNet]="alexnet_accelerator"
NET_DEF[AlexNet]="NETWORK_ALEXNET"
NET_LIBS[AlexNet]=""

NET_RTL[MobileNetV2]="rtl/mobilenetv2_accelerator.v rtl/conv_pe_array.v rtl/pool_layer.v rtl/residual_add.v rtl/depthwise_conv.v"
NET_TOP[MobileNetV2]="mobilenetv2_accelerator"
NET_DEF[MobileNetV2]="NETWORK_MOBILENETV2"
NET_LIBS[MobileNetV2]=""

NET_RTL[GoogLeNet]="rtl/googlenet_accelerator.v rtl/conv_pe_array.v rtl/pool_layer.v"
NET_TOP[GoogLeNet]="googlenet_accelerator"
NET_DEF[GoogLeNet]="NETWORK_GOOGLENET"
NET_LIBS[GoogLeNet]=""

for NET in VGG16 ResNet50 AlexNet MobileNetV2 GoogLeNet; do
    echo
    echo "========== Step 2: 构建 $NET =========="
    OBJ=obj_$(echo $NET | tr 'A-Z' 'a-z')
    rm -rf "$OBJ"
    mkdir -p "$OBJ"
    t0=$(date +%s.%N)
    verilator -Wall --cc -O2 -Wno-fatal --top-module "${NET_TOP[$NET]}" \
        -Mdir "$OBJ" \
        ${NET_RTL[$NET]} \
        --exe tb/tb_generic.cpp tb/dmm_scheduler.cpp tb/dmm_tree_generated.cpp \
               tb/pts_scorer.cpp tb/tl_layers.cpp \
        -CFLAGS "-I$ROOT/tb -D${NET_DEF[$NET]}" \
        --build -j $JOBS > "compile_${NET}.log" 2>&1
    t1=$(date +%s.%N)
    awk -v s="$t0" -v e="$t1" -v n="$NET" 'BEGIN{printf "%s build: %.2f s\n", n, e-s}'
done

echo
echo "========== Step 3: 5 网络 × 4 模式 =========="
printf "\n%-12s %-10s %-5s %-5s %-14s %-14s %-6s\n" \
       "Network" "Mode" "DMM" "PTS" "Total Cycles" "Compute(ms)" "Done"
printf -- "---------------------------------------------------------------------------------\n"

declare -A BASE_CYCLES
for NET in VGG16 ResNet50 AlexNet MobileNetV2 GoogLeNet; do
    OBJ=obj_$(echo $NET | tr 'A-Z' 'a-z')
    case "$NET" in
        VGG16)       EXE=$OBJ/Vvgg16_accelerator ;;
        ResNet50)    EXE=$OBJ/Vresnet50_accelerator ;;
        AlexNet)     EXE=$OBJ/Valexnet_accelerator ;;
        MobileNetV2) EXE=$OBJ/Vmobilenetv2_accelerator ;;
        GoogLeNet)   EXE=$OBJ/Vgooglenet_accelerator ;;
    esac
    for MODE in baseline pts dmm pts_dmm; do
        case "$MODE" in
            baseline) DMM="NO";  PTS="NO"  ;;
            pts)      DMM="NO";  PTS="YES" ;;
            dmm)      DMM="YES"; PTS="NO"  ;;
            pts_dmm)  DMM="YES"; PTS="YES" ;;
        esac
        LOG="sim_${NET}_${MODE}.log"
        VERILATOR_X_MODE="$MODE" timeout 1800 stdbuf -oL -eL "$EXE" > "$LOG" 2>&1 || true
        TC=$(awk '/^Total cycles:/  {print $3}' "$LOG")
        CT=$(awk '/^Compute time:/  {print $3}' "$LOG")
        DN=$(awk '/^Done signal:/   {print $3}' "$LOG")
        printf "%-12s %-10s %-5s %-5s %-14s %-14s %-6s\n" \
               "$NET" "$MODE" "$DMM" "$PTS" "$TC" "$CT" "$DN"
        if [ "$MODE" = "baseline" ]; then BASE_CYCLES[$NET]=$TC; fi
    done
done

echo
echo "========== Step 4: 加速比 =========="
printf "\n%-12s %-10s %-16s\n" "Network" "Mode" "Speedup"
printf -- "------------------------------------------\n"
for NET in VGG16 ResNet50 AlexNet MobileNetV2 GoogLeNet; do
    B=${BASE_CYCLES[$NET]}
    for MODE in pts dmm pts_dmm; do
        T=$(awk '/^Total cycles:/ {print $3}' "sim_${NET}_${MODE}.log")
        awk -v b="$B" -v t="$T" -v n="$NET" -v m="$MODE" \
            'BEGIN{ if (t>0) printf "%-12s %-10s %.2fx\n", n, m, b/t }'
    done
done
