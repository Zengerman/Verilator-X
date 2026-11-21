`timescale 1ns/1ps
//64x64 size PE 
module conv_pe_array #(
    parameter ARRAY_SIZE=64, IN_CH=3, OUT_CH=64,
    parameter FEATURE_SIZE=16, KERNEL=3
)(
    input  wire clk, reset_n, start,
    input  wire [7:0] feature_in  [0:FEATURE_SIZE-1][0:FEATURE_SIZE-1][0:IN_CH-1],
    output reg  [7:0] feature_out [0:FEATURE_SIZE-1][0:FEATURE_SIZE-1][0:OUT_CH-1],
    input  wire        weight_we,
    input  wire [15:0] weight_addr,
    input  wire [7:0]  weight_data,
    output reg         done,
    output reg  [31:0] cycle_count
);
    localparam MAC_TOTAL  = KERNEL*KERNEL*IN_CH;
    localparam TILE_COUNT = (FEATURE_SIZE*FEATURE_SIZE + ARRAY_SIZE - 1)/ARRAY_SIZE;

    reg signed [31:0] acc [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg [7:0] weight_mem [0:OUT_CH-1][0:MAC_TOTAL-1];
    reg [7:0] act_buf [0:ARRAY_SIZE-1][0:IN_CH-1];

    reg [31:0] w_oc, w_idx;
    always @(*) begin
        w_oc  = {16'b0, weight_addr} / MAC_TOTAL;
        w_idx = {16'b0, weight_addr} % MAC_TOTAL;
    end
    always @(posedge clk) if (weight_we && w_oc < OUT_CH)
        weight_mem[w_oc][w_idx] <= weight_data;

    localparam IDLE=2'd0, LOAD=2'd1, COMPUTE=2'd2, WB=2'd3;
    reg [1:0] state;
    reg [31:0] tile_idx, mac_iter, spatial_base;
    integer r, c, pix, oi, oj, ii, jj, ic, kx, ky;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state<=IDLE; done<=0; cycle_count<=0;
            tile_idx<=0; mac_iter<=0; spatial_base<=0;
            for (r=0; r<ARRAY_SIZE; r=r+1)
                for (c=0; c<ARRAY_SIZE; c=c+1) acc[r][c]<=0;
        end else case (state)
            IDLE: begin done<=0;
                if (start) begin state<=LOAD; tile_idx<=0; spatial_base<=0; end
            end
            LOAD: begin
                for (r=0; r<ARRAY_SIZE; r=r+1) begin
                    pix = spatial_base + r;
                    oi  = pix / FEATURE_SIZE;
                    oj  = pix % FEATURE_SIZE;
                    for (c=0; c<IN_CH; c=c+1)
                        act_buf[r][c] <= (oi<FEATURE_SIZE && oj<FEATURE_SIZE)
                                       ? feature_in[oi][oj][c] : 8'd0;
                    for (c=0; c<ARRAY_SIZE; c=c+1) acc[r][c] <= 0;
                end
                mac_iter<=0; state<=COMPUTE; cycle_count<=cycle_count+1;
            end
            COMPUTE: begin
                ic = mac_iter % IN_CH;
                kx = (mac_iter / IN_CH) % KERNEL;
                ky = (mac_iter / IN_CH / KERNEL) % KERNEL;
                for (r=0; r<ARRAY_SIZE; r=r+1) begin
                    pix = spatial_base + r;
                    oi  = pix / FEATURE_SIZE;
                    oj  = pix % FEATURE_SIZE;
                    ii  = oi + kx - 1;
                    jj  = oj + ky - 1;
                    if (ii>=0 && ii<FEATURE_SIZE && jj>=0 && jj<FEATURE_SIZE)
                        for (c=0; c<ARRAY_SIZE; c=c+1)
                            acc[r][c] <= acc[r][c]
                                       + act_buf[r][ic] * weight_mem[c][mac_iter];
                end
                mac_iter<=mac_iter+1; cycle_count<=cycle_count+1;
                if (mac_iter == MAC_TOTAL - 1) state <= WB;
            end
            WB: begin
                for (r=0; r<ARRAY_SIZE; r=r+1) begin
                    pix = spatial_base + r;
                    oi  = pix / FEATURE_SIZE;
                    oj  = pix % FEATURE_SIZE;
                    if (oi<FEATURE_SIZE && oj<FEATURE_SIZE)
                        for (c=0; c<OUT_CH; c=c+1)
                            feature_out[oi][oj][c] <= (acc[r][c]>0) ? acc[r][c][7:0] : 8'd0;
                end
                cycle_count <= cycle_count + 1;
                if (tile_idx == TILE_COUNT - 1) begin state<=IDLE; done<=1; end
                else begin
                    tile_idx<=tile_idx+1; spatial_base<=spatial_base+ARRAY_SIZE;
                    state<=LOAD;
                end
            end
        endcase
    end
endmodule
