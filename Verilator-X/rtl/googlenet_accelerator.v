`timescale 1ns/1ps
module googlenet_accelerator (
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
    input  wire [7:0] dbg_pool1_out [0:7][0:7][0:63],
    input  wire [7:0] dbg_concat_out [0:7][0:7][0:255],   
    output wire       wait_pts,
    output wire [3:0] pts_layer_idx
);
    wire [7:0] buf1 [0:15][0:15][0:63];
    wire [7:0] buf2 [0:7][0:7][0:63];
    wire [7:0] buf2_mux [0:7][0:7][0:63];       // ← wire 声明在 generate 之前
    wire [7:0] buf_b1 [0:7][0:7][0:63];
    wire [7:0] buf_b2a [0:7][0:7][0:31];
    wire [7:0] buf_b2 [0:7][0:7][0:127];
    wire [7:0] buf_b3a [0:7][0:7][0:15];
    wire [7:0] buf_b3 [0:7][0:7][0:31];
    wire [7:0] buf_b4 [0:7][0:7][0:31];
    wire [7:0] buf_concat [0:7][0:7][0:255];
    wire [7:0] buf3 [0:3][0:3][0:255];
    wire [7:0] buf4 [0:3][0:3][0:255];

    wire w1 = weight_we && (weight_addr[23:20]==4'd0);
    wire w2 = weight_we && (weight_addr[23:20]==4'd1);
    wire w3 = weight_we && (weight_addr[23:20]==4'd2);
    wire w4 = weight_we && (weight_addr[23:20]==4'd3);
    wire w5 = weight_we && (weight_addr[23:20]==4'd4);
    wire w6 = weight_we && (weight_addr[23:20]==4'd5);
    wire w7 = weight_we && (weight_addr[23:20]==4'd6);
    wire w8 = weight_we && (weight_addr[23:20]==4'd7);

    wire c1s,p1s,b1s,b2as,b2s,b3as,b3s,b4s,p2s,cfs;
    wire c1d,p1d,b1d,b2ad,b2d,b3ad,b3d,b4d,p2d,cfd;

    // conv1: 3→64, 16×16, K=3
    conv_pe_array #(.IN_CH(3),.OUT_CH(64),.FEATURE_SIZE(16),.KERNEL(3)) conv1 (
        .clk(clk),.reset_n(reset_n),.start(c1s),.feature_in(feature_in),
        .feature_out(buf1),.weight_we(w1),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(c1d),.cycle_count());

    // pool1 输入永远是 buf1（PTS 时 pool1 RTL 不启动，不读它）
    pool_layer #(.CH(64),.IN_SIZE(16),.CYCLES_PER_STEP(32)) pool1 (
        .clk(clk),.reset_n(reset_n),.start(p1s),.feature_in(buf1),
        .feature_out(buf2),.done(p1d),.cycle_count());

    // pool1 输出选择：PTS 用 dbg_pool1_out，否则 buf2
    generate
        for (genvar i=0;i<8;i=i+1)
            for (genvar j=0;j<8;j=j+1)
                for (genvar c=0;c<64;c=c+1)
                    assign buf2_mux[i][j][c] = pts_mode ? dbg_pool1_out[i][j][c] : buf2[i][j][c];
    endgenerate

    // Inception 分支 1: 1×1(64→64)
    conv_pe_array #(.IN_CH(64),.OUT_CH(64),.FEATURE_SIZE(8),.KERNEL(1)) b1 (
        .clk(clk),.reset_n(reset_n),.start(b1s),.feature_in(buf2_mux),
        .feature_out(buf_b1),.weight_we(w2),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(b1d),.cycle_count());

    // 分支 2: 1×1(64→32) → 3×3(32→128)
    conv_pe_array #(.IN_CH(64),.OUT_CH(32),.FEATURE_SIZE(8),.KERNEL(1)) b2a (
        .clk(clk),.reset_n(reset_n),.start(b2as),.feature_in(buf2_mux),
        .feature_out(buf_b2a),.weight_we(w3),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(b2ad),.cycle_count());

    conv_pe_array #(.IN_CH(32),.OUT_CH(128),.FEATURE_SIZE(8),.KERNEL(3)) b2 (
        .clk(clk),.reset_n(reset_n),.start(b2s),.feature_in(buf_b2a),
        .feature_out(buf_b2),.weight_we(w4),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(b2d),.cycle_count());

    // 分支 3: 1×1(64→16) → 5×5(16→32)
    conv_pe_array #(.IN_CH(64),.OUT_CH(16),.FEATURE_SIZE(8),.KERNEL(1)) b3a (
        .clk(clk),.reset_n(reset_n),.start(b3as),.feature_in(buf2_mux),
        .feature_out(buf_b3a),.weight_we(w5),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(b3ad),.cycle_count());

    conv_pe_array #(.IN_CH(16),.OUT_CH(32),.FEATURE_SIZE(8),.KERNEL(5)) b3 (
        .clk(clk),.reset_n(reset_n),.start(b3s),.feature_in(buf_b3a),
        .feature_out(buf_b3),.weight_we(w6),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(b3d),.cycle_count());

    // 分支 4: 1×1(64→32)
    conv_pe_array #(.IN_CH(64),.OUT_CH(32),.FEATURE_SIZE(8),.KERNEL(1)) b4 (
        .clk(clk),.reset_n(reset_n),.start(b4s),.feature_in(buf2_mux),
        .feature_out(buf_b4),.weight_we(w7),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(b4d),.cycle_count());

    // concat: b1(64) + b2(128) + b3(32) + b4(32) = 256
    generate
        for (genvar i=0;i<8;i=i+1)
            for (genvar j=0;j<8;j=j+1) begin : gen_concat
                for (genvar c=0;c<64;c=c+1)  assign buf_concat[i][j][c]     = buf_b1[i][j][c];
                for (genvar c=0;c<128;c=c+1) assign buf_concat[i][j][64+c]  = buf_b2[i][j][c];
                for (genvar c=0;c<32;c=c+1)  assign buf_concat[i][j][192+c] = buf_b3[i][j][c];
                for (genvar c=0;c<32;c=c+1)  assign buf_concat[i][j][224+c] = buf_b4[i][j][c];
            end
    endgenerate

    // pool2 输入直接用 buf_concat（pool2 不做 PTS，走 RTL）
    pool_layer #(.CH(256),.IN_SIZE(8),.CYCLES_PER_STEP(16)) pool2 (
        .clk(clk),.reset_n(reset_n),.start(p2s),.feature_in(buf_concat),
        .feature_out(buf3),.done(p2d),.cycle_count());

    conv_pe_array #(.IN_CH(256),.OUT_CH(256),.FEATURE_SIZE(4),.KERNEL(3)) conv_f (
        .clk(clk),.reset_n(reset_n),.start(cfs),.feature_in(buf3),
        .feature_out(buf4),.weight_we(w8),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(cfd),.cycle_count());
    assign feature_out = buf4;

    // ---- 状态机（去掉 W2，W1 后直接进 B1）----
    localparam IDLE=4'd0,C1=4'd1,W1=4'd2,P1=4'd3,B1=4'd4,B2A=4'd5,B2=4'd6,
               B3A=4'd7,B3=4'd8,B4=4'd9,CONCAT=4'd10,P2=4'd11,CF=4'd12,DONE=4'd13;
    reg [3:0] state;
    reg c1s_r,p1s_r,b1s_r,b2as_r,b2s_r,b3as_r,b3s_r,b4s_r,p2s_r,cfs_r;
    assign c1s=c1s_r;assign p1s=p1s_r;assign b1s=b1s_r;assign b2as=b2as_r;
    assign b2s=b2s_r;assign b3as=b3as_r;assign b3s=b3s_r;assign b4s=b4s_r;
    assign p2s=p2s_r;assign cfs=cfs_r;

    assign wait_pts      = (state == W1);       // 只 pool1
    assign pts_layer_idx = 4'd1;                // 只返回 1

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state<=IDLE; done<=0;
            c1s_r<=0;p1s_r<=0;b1s_r<=0;b2as_r<=0;b2s_r<=0;
            b3as_r<=0;b3s_r<=0;b4s_r<=0;p2s_r<=0;cfs_r<=0;
        end else begin
            c1s_r<=0;p1s_r<=0;b1s_r<=0;b2as_r<=0;b2s_r<=0;
            b3as_r<=0;b3s_r<=0;b4s_r<=0;p2s_r<=0;cfs_r<=0;done<=0;
            case (state)
                IDLE: if (start) begin c1s_r<=1; state<=C1; end
                C1: if (c1d) begin
                    if (pts_mode)      state<=W1;           // PTS：等 tb
                    else begin p1s_r<=1; state<=P1; end     // 非 PTS：跑 RTL
                end
                W1: if (pts_ready) begin b1s_r<=1; state<=B1; end   // 跳过 P1
                P1: if (p1d) begin b1s_r<=1; state<=B1; end
                B1: if (b1d) begin b2as_r<=1; state<=B2A; end
                B2A: if (b2ad) begin b2s_r<=1; state<=B2; end
                B2: if (b2d) begin b3as_r<=1; state<=B3A; end
                B3A: if (b3ad) begin b3s_r<=1; state<=B3; end
                B3: if (b3d) begin b4s_r<=1; state<=B4; end
                B4: if (b4d) state<=CONCAT;
                CONCAT: begin p2s_r<=1; state<=P2; end              // 直接跑 pool2
                P2: if (p2d) begin cfs_r<=1; state<=CF; end
                CF: if (cfd) state<=DONE;
                DONE: begin done<=1; state<=IDLE; end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
