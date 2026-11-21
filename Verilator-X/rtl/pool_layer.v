`timescale 1ns/1ps
//pool_layer 2×2 max pooling
module pool_layer #(
    parameter CH=64, IN_SIZE=16, CYCLES_PER_STEP=32
)(
    input  wire clk, reset_n, start,
    input  wire [7:0] feature_in  [0:IN_SIZE-1][0:IN_SIZE-1][0:CH-1],
    output reg  [7:0] feature_out [0:IN_SIZE/2-1][0:IN_SIZE/2-1][0:CH-1],
    output reg         done,
    output reg  [31:0] cycle_count
);
    localparam OUT_SIZE = IN_SIZE/2;
    localparam TOTAL    = OUT_SIZE*OUT_SIZE*CH;
    localparam IDLE=1'd0, COMPUTE=1'd1;
    reg state;
    reg [31:0] work_idx;
    reg [7:0]  step_cnt;
    integer idx, c, pix, pi, pj;
    reg [7:0] maxv;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state<=IDLE; done<=0; work_idx<=0; step_cnt<=0; cycle_count<=0;
        end else case (state)
            IDLE: begin done<=0;
                if (start) begin state<=COMPUTE; work_idx<=0; step_cnt<=0; end
            end
            COMPUTE: begin
                step_cnt <= step_cnt + 1;
                cycle_count <= cycle_count + 1;
                if (step_cnt == CYCLES_PER_STEP - 1) begin
                    idx = work_idx;
                    c   = idx % CH;
                    pix = idx / CH;
                    pi  = pix / OUT_SIZE;
                    pj  = pix % OUT_SIZE;
                    maxv = feature_in[2*pi][2*pj][c];
                    if (feature_in[2*pi+1][2*pj][c]   > maxv) maxv = feature_in[2*pi+1][2*pj][c];
                    if (feature_in[2*pi][2*pj+1][c]   > maxv) maxv = feature_in[2*pi][2*pj+1][c];
                    if (feature_in[2*pi+1][2*pj+1][c] > maxv) maxv = feature_in[2*pi+1][2*pj+1][c];
                    feature_out[pi][pj][c] <= maxv;
                    step_cnt <= 0;
                    if (work_idx == TOTAL - 1) begin state<=IDLE; done<=1; end
                    else work_idx <= work_idx + 1;
                end
            end
        endcase
    end
endmodule
