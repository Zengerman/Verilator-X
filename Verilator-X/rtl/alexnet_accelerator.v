`timescale 1ns/1ps
module alexnet_accelerator (
    input  wire clk, reset_n, start,
    input  wire [7:0] feature_in [0:15][0:15][0:2],
    input  wire        weight_we,
    input  wire [23:0] weight_addr,
    input  wire [7:0]  weight_data,
    output wire [7:0] feature_out [0:3][0:3][0:255],
    output reg  done,
    // ---- PTS 控制 ----
    input  wire        pts_mode,
    input  wire        pts_ready,
    output wire        wait_pts,
    output wire [3:0]  pts_layer_idx,
    // ---- RTL 输出（tb 读）----
    output wire [7:0] dbg_conv1_out [0:15][0:15][0:63],
    output wire [7:0] dbg_conv2_out [0:7][0:7][0:127],
    // ---- tb 输入（PTS 结果写回）----
    input  wire [7:0] dbg_pool1_out [0:7][0:7][0:63],
    input  wire [7:0] dbg_pool2_out [0:3][0:3][0:127]
);
    wire [7:0] buf1 [0:15][0:15][0:63];
    wire [7:0] buf2 [0:7][0:7][0:63];
    wire [7:0] buf2_mux [0:7][0:7][0:63];
    wire [7:0] buf3 [0:7][0:7][0:127];
    wire [7:0] buf4 [0:3][0:3][0:127];
    wire [7:0] buf4_mux [0:3][0:3][0:127];
    wire [7:0] buf5 [0:3][0:3][0:191];
    wire [7:0] buf6 [0:3][0:3][0:191];
    wire [7:0] buf7 [0:3][0:3][0:255];

    wire w1 = weight_we && (weight_addr[23:20]==4'd0);
    wire w2 = weight_we && (weight_addr[23:20]==4'd1);
    wire w3 = weight_we && (weight_addr[23:20]==4'd2);
    wire w4 = weight_we && (weight_addr[23:20]==4'd3);
    wire w5 = weight_we && (weight_addr[23:20]==4'd4);

    wire c1s,c2s,c3s,c4s,c5s,p1s,p2s;
    wire c1d,c2d,c3d,c4d,c5d,p1d,p2d;

    // ---- conv1: 3→64, 16×16, K=7 ----
    conv_pe_array #(.IN_CH(3), .OUT_CH(64), .FEATURE_SIZE(16), .KERNEL(7)) conv1 (
        .clk(clk), .reset_n(reset_n), .start(c1s), .feature_in(feature_in),
        .feature_out(buf1), .weight_we(w1), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(c1d), .cycle_count());

    assign dbg_conv1_out = buf1;   // RTL 输出给 tb

    // ---- pool1: 16→8（PTS 可跳过）----
    // pool1 的 RTL 输入永远是 conv1 输出 buf1；
    // PTS 模式下 pool1 的 RTL 不启动，由 tb 用 dbg_conv1_out 算完写 dbg_pool1_out。
    pool_layer #(.CH(64), .IN_SIZE(16), .CYCLES_PER_STEP(32)) pool1 (
        .clk(clk), .reset_n(reset_n), .start(p1s), .feature_in(buf1),
        .feature_out(buf2), .done(p1d), .cycle_count());

    // conv2 输入：非 PTS 用 pool1 RTL 输出 buf2；PTS 用 tb 写的 dbg_pool1_out
    generate
        for (genvar i=0;i<8;i=i+1)
            for (genvar j=0;j<8;j=j+1)
                for (genvar c=0;c<64;c=c+1)
                    assign buf2_mux[i][j][c] = pts_mode ? dbg_pool1_out[i][j][c] : buf2[i][j][c];
    endgenerate

    // ---- conv2: 64→128, 8×8, K=5 ----
    conv_pe_array #(.IN_CH(64), .OUT_CH(128), .FEATURE_SIZE(8), .KERNEL(5)) conv2 (
        .clk(clk), .reset_n(reset_n), .start(c2s), .feature_in(buf2_mux),
        .feature_out(buf3), .weight_we(w2), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(c2d), .cycle_count());

    assign dbg_conv2_out = buf3;

    // ---- pool2: 8→4（PTS 可跳过）----
    pool_layer #(.CH(128), .IN_SIZE(8), .CYCLES_PER_STEP(16)) pool2 (
        .clk(clk), .reset_n(reset_n), .start(p2s), .feature_in(buf3),
        .feature_out(buf4), .done(p2d), .cycle_count());

    generate
        for (genvar i=0;i<4;i=i+1)
            for (genvar j=0;j<4;j=j+1)
                for (genvar c=0;c<128;c=c+1)
                    assign buf4_mux[i][j][c] = pts_mode ? dbg_pool2_out[i][j][c] : buf4[i][j][c];
    endgenerate

    // ---- conv3~conv5 ----
    conv_pe_array #(.IN_CH(128), .OUT_CH(192), .FEATURE_SIZE(4), .KERNEL(3)) conv3 (
        .clk(clk), .reset_n(reset_n), .start(c3s), .feature_in(buf4_mux),
        .feature_out(buf5), .weight_we(w3), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(c3d), .cycle_count());

    conv_pe_array #(.IN_CH(192), .OUT_CH(192), .FEATURE_SIZE(4), .KERNEL(3)) conv4 (
        .clk(clk), .reset_n(reset_n), .start(c4s), .feature_in(buf5),
        .feature_out(buf6), .weight_we(w4), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(c4d), .cycle_count());

    conv_pe_array #(.IN_CH(192), .OUT_CH(256), .FEATURE_SIZE(4), .KERNEL(3)) conv5 (
        .clk(clk), .reset_n(reset_n), .start(c5s), .feature_in(buf6),
        .feature_out(buf7), .weight_we(w5), .weight_addr(weight_addr[15:0]),
        .weight_data(weight_data), .done(c5d), .cycle_count());

    assign feature_out = buf7;

    // ---- 状态机 ----
    localparam IDLE=4'd0, C1=4'd1, W1=4'd2, P1=4'd3,
               C2=4'd4, W2=4'd5, P2=4'd6,
               C3=4'd7, C4=4'd8, C5=4'd9, DONE=4'd10;
    reg [3:0] state;
    reg c1s_r,c2s_r,c3s_r,c4s_r,c5s_r,p1s_r,p2s_r;

    assign c1s=c1s_r; assign c2s=c2s_r; assign c3s=c3s_r;
    assign c4s=c4s_r; assign c5s=c5s_r;
    assign p1s=p1s_r; assign p2s=p2s_r;

    assign wait_pts = (state==W1) || (state==W2);
    assign pts_layer_idx = (state==W1) ? 4'd1 : (state==W2) ? 4'd3 : 4'd0;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state<=IDLE; done<=0;
            c1s_r<=0;c2s_r<=0;c3s_r<=0;c4s_r<=0;c5s_r<=0;p1s_r<=0;p2s_r<=0;
        end else begin
            c1s_r<=0;c2s_r<=0;c3s_r<=0;c4s_r<=0;c5s_r<=0;p1s_r<=0;p2s_r<=0;done<=0;
            case (state)
                IDLE: if (start) begin c1s_r<=1; state<=C1; end
                C1: if (c1d) begin
                    if (pts_mode)      state<=W1;            // PTS：等 tb 算 pool1
                    else begin p1s_r<=1; state<=P1; end      // 非 PTS：跑 RTL pool1
                end
                W1: if (pts_ready) begin c2s_r<=1; state<=C2; end   // PTS：直接进 conv2
                P1: if (p1d) begin c2s_r<=1; state<=C2; end
                C2: if (c2d) begin
                    if (pts_mode)      state<=W2;
                    else begin p2s_r<=1; state<=P2; end
                end
                W2: if (pts_ready) begin c3s_r<=1; state<=C3; end   // PTS：直接进 conv3
                P2: if (p2d) begin c3s_r<=1; state<=C3; end
                C3: if (c3d) begin c4s_r<=1; state<=C4; end
                C4: if (c4d) begin c5s_r<=1; state<=C5; end
                C5: if (c5d) state<=DONE;
                DONE: begin done<=1; state<=IDLE; end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
