`timescale 1ns/1ps
module mobilenetv2_accelerator (
    input  wire clk, reset_n, start,
    input  wire [7:0] feature_in [0:15][0:15][0:2],
    input  wire        weight_we,
    input  wire [23:0] weight_addr,
    input  wire [7:0]  weight_data,
    output wire [7:0] feature_out [0:3][0:3][0:255],
    output reg  done,
    input  wire        pts_mode,
    input  wire        pts_ready,
    input  wire [7:0] dbg_conv1_out [0:15][0:15][0:63],
    input  wire [7:0] dbg_pw1_out   [0:15][0:15][0:63],
    input  wire [7:0] dbg_res1_out  [0:15][0:15][0:63],
    input  wire [7:0] dbg_pool1_out [0:7][0:7][0:63],
    input  wire [7:0] dbg_conv2_out [0:7][0:7][0:127],
    input  wire [7:0] dbg_pw2_out   [0:7][0:7][0:127],
    input  wire [7:0] dbg_res2_out  [0:7][0:7][0:127],
    input  wire [7:0] dbg_pool2_out [0:3][0:3][0:127],
    output wire       wait_pts,
    output wire [3:0] pts_layer_idx
);
    // ---- 每层独立 buffer，避免多驱动 ----
    wire [7:0] b_conv1 [0:15][0:15][0:63];
    wire [7:0] b_dw1   [0:15][0:15][0:63];
    wire [7:0] b_pw1   [0:15][0:15][0:63];
    wire [7:0] b_res1  [0:15][0:15][0:63];
    wire [7:0] b_pool1 [0:7][0:7][0:63];
    wire [7:0] b_conv2 [0:7][0:7][0:127];
    wire [7:0] b_dw2   [0:7][0:7][0:127];
    wire [7:0] b_pw2   [0:7][0:7][0:127];
    wire [7:0] b_res2  [0:7][0:7][0:127];
    wire [7:0] b_pool2 [0:3][0:3][0:127];
    wire [7:0] b_conv3 [0:3][0:3][0:255];

    // ---- PTS 多路 ----
    wire [7:0] b_conv1_mux [0:15][0:15][0:63];
    wire [7:0] b_pw1_mux   [0:15][0:15][0:63];
    wire [7:0] b_res1_mux  [0:15][0:15][0:63];
    wire [7:0] b_pool1_mux [0:7][0:7][0:63];
    wire [7:0] b_conv2_mux [0:7][0:7][0:127];
    wire [7:0] b_pw2_mux   [0:7][0:7][0:127];
    wire [7:0] b_res2_mux  [0:7][0:7][0:127];
    wire [7:0] b_pool2_mux [0:3][0:3][0:127];

    genvar gi, gj, gc;
    generate
        for (gi=0; gi<16; gi=gi+1) for (gj=0; gj<16; gj=gj+1) for (gc=0; gc<64; gc=gc+1) begin : gen_mux1
            assign b_conv1_mux[gi][gj][gc] = pts_mode ? dbg_conv1_out[gi][gj][gc] : b_conv1[gi][gj][gc];
            assign b_pw1_mux[gi][gj][gc]   = pts_mode ? dbg_pw1_out[gi][gj][gc]   : b_pw1[gi][gj][gc];
            assign b_res1_mux[gi][gj][gc]  = pts_mode ? dbg_res1_out[gi][gj][gc]  : b_res1[gi][gj][gc];
        end
        for (gi=0; gi<8; gi=gi+1) for (gj=0; gj<8; gj=gj+1) begin : gen_mux2
            for (gc=0; gc<64; gc=gc+1)
                assign b_pool1_mux[gi][gj][gc] = pts_mode ? dbg_pool1_out[gi][gj][gc] : b_pool1[gi][gj][gc];
            for (gc=0; gc<128; gc=gc+1) begin
                assign b_conv2_mux[gi][gj][gc] = pts_mode ? dbg_conv2_out[gi][gj][gc] : b_conv2[gi][gj][gc];
                assign b_pw2_mux[gi][gj][gc]   = pts_mode ? dbg_pw2_out[gi][gj][gc]   : b_pw2[gi][gj][gc];
                assign b_res2_mux[gi][gj][gc]  = pts_mode ? dbg_res2_out[gi][gj][gc]  : b_res2[gi][gj][gc];
            end
        end
        for (gi=0; gi<4; gi=gi+1) for (gj=0; gj<4; gj=gj+1) for (gc=0; gc<128; gc=gc+1) begin : gen_mux3
            assign b_pool2_mux[gi][gj][gc] = pts_mode ? dbg_pool2_out[gi][gj][gc] : b_pool2[gi][gj][gc];
        end
    endgenerate

    // ---- 权重分发 ----
    wire w1 = weight_we && (weight_addr[23:20]==4'd0);  // conv1
    wire w2 = weight_we && (weight_addr[23:20]==4'd1);  // dw1
    wire w3 = weight_we && (weight_addr[23:20]==4'd2);  // pw1
    wire w4 = weight_we && (weight_addr[23:20]==4'd3);  // conv2
    wire w5 = weight_we && (weight_addr[23:20]==4'd4);  // dw2
    wire w6 = weight_we && (weight_addr[23:20]==4'd5);  // pw2
    wire w7 = weight_we && (weight_addr[23:20]==4'd6);  // conv3

    wire c1s, dw1s, pw1s, r1s, p1s, c2s, dw2s, pw2s, r2s, p2s, c3s;
    wire c1d, dw1d, pw1d, r1d, p1d, c2d, dw2d, pw2d, r2d, p2d, c3d;

    // 1) conv1: 3 → 64, 16×16, K=3
    conv_pe_array #(.IN_CH(3), .OUT_CH(64), .FEATURE_SIZE(16), .KERNEL(3)) conv1 (
        .clk(clk), .reset_n(reset_n), .start(c1s), .feature_in(feature_in),
        .feature_out(b_conv1), .weight_we(w1), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(c1d), .cycle_count());

    // 2) dw1: 64 ch, 16×16, K=3（PTS 跳过）
    depthwise_conv #(.CH(64), .FEATURE_SIZE(16), .KERNEL(3)) dw1 (
        .clk(clk), .reset_n(reset_n), .start(dw1s), .feature_in(b_conv1_mux),
        .feature_out(b_dw1), .weight_we(w2), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(dw1d), .cycle_count());

    // 3) pw1: 64 → 64, 16×16, K=1
    conv_pe_array #(.IN_CH(64), .OUT_CH(64), .FEATURE_SIZE(16), .KERNEL(1)) pw1 (
        .clk(clk), .reset_n(reset_n), .start(pw1s), .feature_in(b_dw1),
        .feature_out(b_pw1), .weight_we(w3), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(pw1d), .cycle_count());

    // 4) res1: b_pw1 + b_conv1（PTS 跳过）
    residual_add #(.CH(64), .SIZE(16), .CYCLES_PER_STEP(16)) res1 (
        .clk(clk), .reset_n(reset_n), .start(r1s), .a(b_pw1_mux), .b(b_conv1),
        .y(b_res1), .done(r1d), .cycle_count());

    // 5) pool1: 16 → 8（PTS 跳过）
    pool_layer #(.CH(64), .IN_SIZE(16), .CYCLES_PER_STEP(32)) pool1 (
        .clk(clk), .reset_n(reset_n), .start(p1s), .feature_in(b_res1_mux),
        .feature_out(b_pool1), .done(p1d), .cycle_count());

    // 6) conv2: 64 → 128, 8×8, K=3
    conv_pe_array #(.IN_CH(64), .OUT_CH(128), .FEATURE_SIZE(8), .KERNEL(3)) conv2 (
        .clk(clk), .reset_n(reset_n), .start(c2s), .feature_in(b_pool1_mux),
        .feature_out(b_conv2), .weight_we(w4), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(c2d), .cycle_count());

    // 7) dw2: 128 ch, 8×8, K=3（PTS 跳过）
    depthwise_conv #(.CH(128), .FEATURE_SIZE(8), .KERNEL(3)) dw2 (
        .clk(clk), .reset_n(reset_n), .start(dw2s), .feature_in(b_conv2_mux),
        .feature_out(b_dw2), .weight_we(w5), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(dw2d), .cycle_count());

    // 8) pw2: 128 → 128, 8×8, K=1
    conv_pe_array #(.IN_CH(128), .OUT_CH(128), .FEATURE_SIZE(8), .KERNEL(1)) pw2 (
        .clk(clk), .reset_n(reset_n), .start(pw2s), .feature_in(b_dw2),
        .feature_out(b_pw2), .weight_we(w6), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(pw2d), .cycle_count());

    // 9) res2: b_pw2 + b_conv2（PTS 跳过）
    residual_add #(.CH(128), .SIZE(8), .CYCLES_PER_STEP(16)) res2 (
        .clk(clk), .reset_n(reset_n), .start(r2s), .a(b_pw2_mux), .b(b_conv2),
        .y(b_res2), .done(r2d), .cycle_count());

    // 10) pool2: 8 → 4（PTS 跳过）
    pool_layer #(.CH(128), .IN_SIZE(8), .CYCLES_PER_STEP(16)) pool2 (
        .clk(clk), .reset_n(reset_n), .start(p2s), .feature_in(b_res2_mux),
        .feature_out(b_pool2), .done(p2d), .cycle_count());

    // 11) conv3: 128 → 256, 4×4, K=3
    conv_pe_array #(.IN_CH(128), .OUT_CH(256), .FEATURE_SIZE(4), .KERNEL(3)) conv3 (
        .clk(clk), .reset_n(reset_n), .start(c3s), .feature_in(b_pool2_mux),
        .feature_out(b_conv3), .weight_we(w7), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(c3d), .cycle_count());

    assign feature_out = b_conv3;

    // ---- 状态机 ----
    localparam S_IDLE=4'd0, S_C1=4'd1, S_DW1=4'd2, S_PW1=4'd3, S_R1=4'd4,
               S_P1=4'd5,  S_C2=4'd6, S_DW2=4'd7, S_PW2=4'd8, S_R2=4'd9,
               S_P2=4'd10, S_C3=4'd11, S_DONE=4'd12;
    reg [3:0] state;
    reg c1s_r,dw1s_r,pw1s_r,r1s_r,p1s_r,c2s_r,dw2s_r,pw2s_r,r2s_r,p2s_r,c3s_r;
    assign c1s=c1s_r; assign dw1s=dw1s_r; assign pw1s=pw1s_r; assign r1s=r1s_r;
    assign p1s=p1s_r; assign c2s=c2s_r; assign dw2s=dw2s_r; assign pw2s=pw2s_r;
    assign r2s=r2s_r; assign p2s=p2s_r; assign c3s=c3s_r;

    // wait_pts 覆盖所有 PTS 可跳过的层：dw1, pw1(其实 pw 不会被跳过), res1, pool1, dw2, res2, pool2
    // 简化：只让 pool1, pool2, res1, res2 走 PTS，dw 和 pw 走 RTL
    assign wait_pts =
        (state==S_R1) || (state==S_P1) || (state==S_R2) || (state==S_P2);
    assign pts_layer_idx =
        (state==S_R1) ? 4'd3 :   // res1
        (state==S_P1) ? 4'd4 :   // pool1
        (state==S_R2) ? 4'd7 :   // res2
        (state==S_P2) ? 4'd8 :   // pool2
        4'd0;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= S_IDLE; done <= 1'b0;
            c1s_r<=0; dw1s_r<=0; pw1s_r<=0; r1s_r<=0; p1s_r<=0;
            c2s_r<=0; dw2s_r<=0; pw2s_r<=0; r2s_r<=0; p2s_r<=0; c3s_r<=0;
        end else begin
            c1s_r<=0; dw1s_r<=0; pw1s_r<=0; r1s_r<=0; p1s_r<=0;
            c2s_r<=0; dw2s_r<=0; pw2s_r<=0; r2s_r<=0; p2s_r<=0; c3s_r<=0;
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin c1s_r<=1; state<=S_C1; end
                S_C1:  if (c1d)   begin dw1s_r<=1; state<=S_DW1; end
                S_DW1: if (dw1d)  begin pw1s_r<=1; state<=S_PW1; end
                S_PW1: if (pw1d)  begin
                    if (pts_mode) state<=S_R1;   // 等 PTS res1
                    else begin r1s_r<=1; state<=S_R1; end
                end
                S_R1: begin
                    if (pts_mode) begin
                        if (pts_ready) begin p1s_r<=1; state<=S_P1; end
                    end else begin
                        if (r1d) begin p1s_r<=1; state<=S_P1; end
                    end
                end
                S_P1: begin
                    if (pts_mode) begin
                        if (pts_ready) begin c2s_r<=1; state<=S_C2; end
                    end else begin
                        if (p1d) begin c2s_r<=1; state<=S_C2; end
                    end
                end
                S_C2:  if (c2d)  begin dw2s_r<=1; state<=S_DW2; end
                S_DW2: if (dw2d) begin pw2s_r<=1; state<=S_PW2; end
                S_PW2: if (pw2d) begin
                    if (pts_mode) state<=S_R2;
                    else begin r2s_r<=1; state<=S_R2; end
                end
                S_R2: begin
                    if (pts_mode) begin
                        if (pts_ready) begin p2s_r<=1; state<=S_P2; end
                    end else begin
                        if (r2d) begin p2s_r<=1; state<=S_P2; end
                    end
                end
                S_P2: begin
                    if (pts_mode) begin
                        if (pts_ready) begin c3s_r<=1; state<=S_C3; end
                    end else begin
                        if (p2d) begin c3s_r<=1; state<=S_C3; end
                    end
                end
                S_C3:  if (c3d)  state<=S_DONE;
                S_DONE: begin done<=1'b1; state<=S_IDLE; end
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule
