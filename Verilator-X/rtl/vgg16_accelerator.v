`timescale 1ns/1ps
module vgg16_accelerator (
    input  wire clk, reset_n, start,
    input  wire [7:0] feature_in [0:15][0:15][0:2],
    input  wire        weight_we,
    input  wire [23:0] weight_addr,       // [23:20]=layer_id
    input  wire [7:0]  weight_data,
    output wire [7:0] feature_out [0:3][0:3][0:63],
    output reg  done,
    // PTS
    input  wire        pts_mode,
    input  wire        pts_ready,
    input  wire [7:0]  dbg_pool1_out [0:7][0:7][0:63],
    input  wire [7:0]  dbg_pool2_out [0:3][0:3][0:63],
    output wire [7:0]  dbg_conv1_out [0:15][0:15][0:63],
    output wire [7:0]  dbg_conv2_out [0:7][0:7][0:63],
    output wire        wait_pts,
    output wire [3:0]  pts_layer_idx
);
    wire [7:0] buf1 [0:15][0:15][0:63];
    wire [7:0] buf2 [0:7][0:7][0:63];
    wire [7:0] buf2_mux [0:7][0:7][0:63];
    wire [7:0] buf3 [0:7][0:7][0:63];
    wire [7:0] buf4 [0:3][0:3][0:63];
    wire [7:0] buf4_mux [0:3][0:3][0:63];
    wire [7:0] buf5 [0:3][0:3][0:63];

    wire w_c1 = weight_we && (weight_addr[23:20]==4'd0);
    wire w_c2 = weight_we && (weight_addr[23:20]==4'd1);
    wire w_c3 = weight_we && (weight_addr[23:20]==4'd2);

    wire c1s, c2s, c3s, p1s, p2s;
    wire c1d, c2d, c3d, p1d, p2d;
    wire [31:0] cc1, cc2;

    conv_pe_array #(.IN_CH(3),  .OUT_CH(64), .FEATURE_SIZE(16)) conv1 (
        .clk(clk),.reset_n(reset_n),.start(c1s),
        .feature_in(feature_in),.feature_out(buf1),
        .weight_we(w_c1),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(c1d),.cycle_count(cc1));
    assign dbg_conv1_out = buf1;

    pool_layer #(.CH(64),.IN_SIZE(16),.CYCLES_PER_STEP(32)) pool1 (
        .clk(clk),.reset_n(reset_n),.start(p1s),
        .feature_in(buf1),.feature_out(buf2),.done(p1d),.cycle_count());

    generate
        for (genvar i=0;i<8;i=i+1)
            for (genvar j=0;j<8;j=j+1)
                for (genvar c=0;c<64;c=c+1)
                    assign buf2_mux[i][j][c] = pts_mode ? dbg_pool1_out[i][j][c] : buf2[i][j][c];
    endgenerate

    conv_pe_array #(.IN_CH(64),.OUT_CH(64),.FEATURE_SIZE(8)) conv2 (
        .clk(clk),.reset_n(reset_n),.start(c2s),
        .feature_in(buf2_mux),.feature_out(buf3),
        .weight_we(w_c2),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(c2d),.cycle_count(cc2));
    assign dbg_conv2_out = buf3;

    pool_layer #(.CH(64),.IN_SIZE(8),.CYCLES_PER_STEP(16)) pool2 (
        .clk(clk),.reset_n(reset_n),.start(p2s),
        .feature_in(buf3),.feature_out(buf4),.done(p2d),.cycle_count());

    generate
        for (genvar i=0;i<4;i=i+1)
            for (genvar j=0;j<4;j=j+1)
                for (genvar c=0;c<64;c=c+1)
                    assign buf4_mux[i][j][c] = pts_mode ? dbg_pool2_out[i][j][c] : buf4[i][j][c];
    endgenerate

    conv_pe_array #(.IN_CH(64),.OUT_CH(64),.FEATURE_SIZE(4)) conv3 (
        .clk(clk),.reset_n(reset_n),.start(c3s),
        .feature_in(buf4_mux),.feature_out(buf5),
        .weight_we(w_c3),.weight_addr(weight_addr[15:0]),
        .weight_data(weight_data),.done(c3d),.cycle_count());
    assign feature_out = buf5;

    localparam IDLE=4'd0, CONV1=4'd1, POOL1=4'd2, WAIT1=4'd3,
               CONV2=4'd4, POOL2=4'd5, WAIT2=4'd6, CONV3=4'd7, DONE=4'd8;
    reg [3:0] state;
    reg c1s_r, c2s_r, c3s_r, p1s_r, p2s_r;
    assign c1s=c1s_r; assign c2s=c2s_r; assign c3s=c3s_r;
    assign p1s=p1s_r; assign p2s=p2s_r;
    assign wait_pts = (state==WAIT1) || (state==WAIT2);
    assign pts_layer_idx = (state==WAIT1) ? 4'd1 : (state==WAIT2) ? 4'd3 : 4'd0;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin state<=IDLE; done<=0;
            c1s_r<=0;c2s_r<=0;c3s_r<=0;p1s_r<=0;p2s_r<=0;
        end else begin
            c1s_r<=0;c2s_r<=0;c3s_r<=0;p1s_r<=0;p2s_r<=0;done<=0;
            case (state)
                IDLE: if (start) begin c1s_r<=1; state<=CONV1; end
                CONV1: if (c1d) begin
                    if (pts_mode) state<=WAIT1;
                    else begin p1s_r<=1; state<=POOL1; end
                end
                POOL1: if (p1d) begin c2s_r<=1; state<=CONV2; end
                WAIT1: if (pts_ready) begin c2s_r<=1; state<=CONV2; end
                CONV2: if (c2d) begin
                    if (pts_mode) state<=WAIT2;
                    else begin p2s_r<=1; state<=POOL2; end
                end
                POOL2: if (p2d) begin c3s_r<=1; state<=CONV3; end
                WAIT2: if (pts_ready) begin c3s_r<=1; state<=CONV3; end
                CONV3: if (c3d) state<=DONE;
                DONE: begin done<=1; state<=IDLE; end
            endcase
        end
    end
endmodule
