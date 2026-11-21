`timescale 1ns/1ps
module resnet50_accelerator (
    input  wire clk, reset_n, start,
    input  wire [7:0] feature_in [0:15][0:15][0:2],
    input  wire        weight_we,
    input  wire [23:0] weight_addr,
    input  wire [7:0]  weight_data,
    output wire [7:0] feature_out [0:3][0:3][0:63],
    output reg  done,
    input  wire        pts_mode,
    input  wire        pts_ready,
    input  wire [7:0]  dbg_res1_out  [0:15][0:15][0:63],
    input  wire [7:0]  dbg_pool1_out [0:7][0:7][0:63],
    input  wire [7:0]  dbg_res2_out  [0:7][0:7][0:63],
    input  wire [7:0]  dbg_pool2_out [0:3][0:3][0:63],
    output wire [7:0]  dbg_conv1_out [0:15][0:15][0:63],
    output wire [7:0]  dbg_conv2_out [0:15][0:15][0:63],
    output wire [7:0]  dbg_conv3_out [0:7][0:7][0:63],
    output wire [7:0]  dbg_conv4_out [0:7][0:7][0:63],
    output wire        wait_pts,
    output wire [3:0]  pts_layer_idx
);
    wire [7:0] buf1 [0:15][0:15][0:63];
    wire [7:0] buf2 [0:15][0:15][0:63];
    wire [7:0] buf3 [0:15][0:15][0:63];
    wire [7:0] buf3_mux [0:15][0:15][0:63];
    wire [7:0] buf4 [0:7][0:7][0:63];
    wire [7:0] buf5 [0:7][0:7][0:63];
    wire [7:0] buf6 [0:7][0:7][0:63];
    wire [7:0] buf7 [0:7][0:7][0:63];
    wire [7:0] buf7_mux [0:7][0:7][0:63];
    wire [7:0] buf8 [0:3][0:3][0:63];
    wire [7:0] buf9 [0:3][0:3][0:63];
    wire [7:0] buf4_mux [0:7][0:7][0:63];
    wire [7:0] buf8_mux [0:3][0:3][0:63];

    wire w1 = weight_we && (weight_addr[23:20]==4'd0);
    wire w2 = weight_we && (weight_addr[23:20]==4'd1);
    wire w3 = weight_we && (weight_addr[23:20]==4'd2);
    wire w4 = weight_we && (weight_addr[23:20]==4'd3);
    wire w5 = weight_we && (weight_addr[23:20]==4'd4);

    wire c1s,c2s,c3s,c4s,c5s,r1s,r2s,p1s,p2s;
    wire c1d,c2d,c3d,c4d,c5d,r1d,r2d,p1d,p2d;

    conv_pe_array #(.IN_CH(3),.OUT_CH(64),.FEATURE_SIZE(16)) conv1 (
        .clk(clk),.reset_n(reset_n),.start(c1s),
        .feature_in(feature_in),.feature_out(buf1),
        .weight_we(w1),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(c1d),.cycle_count());
    assign dbg_conv1_out = buf1;

    conv_pe_array #(.IN_CH(64),.OUT_CH(64),.FEATURE_SIZE(16)) conv2 (
        .clk(clk),.reset_n(reset_n),.start(c2s),
        .feature_in(buf1),.feature_out(buf2),
        .weight_we(w2),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(c2d),.cycle_count());
    assign dbg_conv2_out = buf2;

    residual_add #(.CH(64),.SIZE(16),.CYCLES_PER_STEP(16)) res1 (
        .clk(clk),.reset_n(reset_n),.start(r1s),
        .a(buf2),.b(buf1),.y(buf3),.done(r1d),.cycle_count());

    generate
        for (genvar i=0;i<16;i=i+1)
            for (genvar j=0;j<16;j=j+1)
                for (genvar c=0;c<64;c=c+1)
                    assign buf3_mux[i][j][c] = pts_mode ? dbg_res1_out[i][j][c] : buf3[i][j][c];
    endgenerate

    pool_layer #(.CH(64),.IN_SIZE(16),.CYCLES_PER_STEP(32)) pool1 (
        .clk(clk),.reset_n(reset_n),.start(p1s),
        .feature_in(buf3_mux),.feature_out(buf4),.done(p1d),.cycle_count());

    generate
        for (genvar i=0;i<8;i=i+1)
            for (genvar j=0;j<8;j=j+1)
                for (genvar c=0;c<64;c=c+1)
                    assign buf4_mux[i][j][c] = pts_mode ? dbg_pool1_out[i][j][c] : buf4[i][j][c];
    endgenerate

    conv_pe_array #(.IN_CH(64),.OUT_CH(64),.FEATURE_SIZE(8)) conv3 (
        .clk(clk),.reset_n(reset_n),.start(c3s),
        .feature_in(buf4_mux),.feature_out(buf5),
        .weight_we(w3),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(c3d),.cycle_count());
    assign dbg_conv3_out = buf5;

    conv_pe_array #(.IN_CH(64),.OUT_CH(64),.FEATURE_SIZE(8)) conv4 (
        .clk(clk),.reset_n(reset_n),.start(c4s),
        .feature_in(buf5),.feature_out(buf6),
        .weight_we(w4),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(c4d),.cycle_count());
    assign dbg_conv4_out = buf6;

    residual_add #(.CH(64),.SIZE(8),.CYCLES_PER_STEP(16)) res2 (
        .clk(clk),.reset_n(reset_n),.start(r2s),
        .a(buf6),.b(buf5),.y(buf7),.done(r2d),.cycle_count());

    generate
        for (genvar i=0;i<8;i=i+1)
            for (genvar j=0;j<8;j=j+1)
                for (genvar c=0;c<64;c=c+1)
                    assign buf7_mux[i][j][c] = pts_mode ? dbg_res2_out[i][j][c] : buf7[i][j][c];
    endgenerate

    pool_layer #(.CH(64),.IN_SIZE(8),.CYCLES_PER_STEP(16)) pool2 (
        .clk(clk),.reset_n(reset_n),.start(p2s),
        .feature_in(buf7_mux),.feature_out(buf8),.done(p2d),.cycle_count());

    generate
        for (genvar i=0;i<4;i=i+1)
            for (genvar j=0;j<4;j=j+1)
                for (genvar c=0;c<64;c=c+1)
                    assign buf8_mux[i][j][c] = pts_mode ? dbg_pool2_out[i][j][c] : buf8[i][j][c];
    endgenerate

    conv_pe_array #(.IN_CH(64),.OUT_CH(64),.FEATURE_SIZE(4)) conv5 (
        .clk(clk),.reset_n(reset_n),.start(c5s),
        .feature_in(buf8_mux),.feature_out(buf9),
        .weight_we(w5),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(c5d),.cycle_count());
    assign feature_out = buf9;

    localparam S_IDLE=4'd0, S_C1=4'd1, S_C2=4'd2, S_R1=4'd3, S_W1=4'd4,
               S_P1=4'd5, S_W1b=4'd6, S_C3=4'd7, S_C4=4'd8, S_R2=4'd9, S_W2=4'd10,
               S_P2=4'd11, S_W2b=4'd12, S_C5=4'd13, S_DONE=4'd14;
    reg [3:0] state;
    reg c1s_r,c2s_r,c3s_r,c4s_r,c5s_r,r1s_r,r2s_r,p1s_r,p2s_r;
    assign c1s=c1s_r; assign c2s=c2s_r; assign c3s=c3s_r;
    assign c4s=c4s_r; assign c5s=c5s_r;
    assign r1s=r1s_r; assign r2s=r2s_r;
    assign p1s=p1s_r; assign p2s=p2s_r;
    assign wait_pts = (state==S_W1)||(state==S_W1b)||(state==S_W2)||(state==S_W2b);
    assign pts_layer_idx =
        (state==S_W1) ? 4'd2 :
        (state==S_W1b)? 4'd3 :
        (state==S_W2) ? 4'd6 :
        (state==S_W2b)? 4'd7 : 4'd0;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state<=S_IDLE; done<=0;
            c1s_r<=0;c2s_r<=0;c3s_r<=0;c4s_r<=0;c5s_r<=0;
            r1s_r<=0;r2s_r<=0;p1s_r<=0;p2s_r<=0;
        end else begin
            c1s_r<=0;c2s_r<=0;c3s_r<=0;c4s_r<=0;c5s_r<=0;
            r1s_r<=0;r2s_r<=0;p1s_r<=0;p2s_r<=0;done<=0;
            case (state)
                S_IDLE: if (start) begin c1s_r<=1; state<=S_C1; end
                S_C1: if (c1d) begin c2s_r<=1; state<=S_C2; end
                S_C2: if (c2d) begin
                    if (pts_mode) state<=S_W1;
                    else begin r1s_r<=1; state<=S_R1; end
                end
                S_R1: if (r1d) begin
                    if (pts_mode) state<=S_W1b;
                    else begin p1s_r<=1; state<=S_P1; end
                end
                S_W1: if (pts_ready) begin
                    if (pts_mode && !p1s_r) begin p1s_r<=1; state<=S_P1; end
                    else begin p1s_r<=1; state<=S_P1; end
                end
                S_W1b: if (pts_ready) begin p1s_r<=1; state<=S_P1; end
                S_P1: if (p1d) begin c3s_r<=1; state<=S_C3; end
                S_C3: if (c3d) begin c4s_r<=1; state<=S_C4; end
                S_C4: if (c4d) begin
                    if (pts_mode) state<=S_W2;
                    else begin r2s_r<=1; state<=S_R2; end
                end
                S_R2: if (r2d) begin
                    if (pts_mode) state<=S_W2b;
                    else begin p2s_r<=1; state<=S_P2; end
                end
                S_W2: if (pts_ready) begin p2s_r<=1; state<=S_P2; end
                S_W2b: if (pts_ready) begin p2s_r<=1; state<=S_P2; end
                S_P2: if (p2d) begin c5s_r<=1; state<=S_C5; end
                S_C5: if (c5d) state<=S_DONE;
                S_DONE: begin done<=1; state<=S_IDLE; end
            endcase
        end
    end
endmodule
