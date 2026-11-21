//残差相加
`timescale 1ns/1ps
module residual_add #(
    parameter CH=64, SIZE=16, CYCLES_PER_STEP=16
)(
    input  wire clk, reset_n, start,
    input  wire [7:0] a [0:SIZE-1][0:SIZE-1][0:CH-1],
    input  wire [7:0] b [0:SIZE-1][0:SIZE-1][0:CH-1],
    output reg  [7:0] y [0:SIZE-1][0:SIZE-1][0:CH-1],
    output reg         done,
    output reg  [31:0] cycle_count
);
    localparam TOTAL = SIZE*SIZE*CH;
    localparam IDLE=1'd0, COMPUTE=1'd1;
    reg state;
    reg [31:0] work_idx;
    reg [7:0]  step_cnt;
    integer idx, c, pix, pi, pj;

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
                    pi  = pix / SIZE;
                    pj  = pix % SIZE;
                    y[pi][pj][c] <= (a[pi][pj][c] + b[pi][pj][c]) & 8'hFF;
                    step_cnt <= 0;
                    if (work_idx == TOTAL - 1) begin state<=IDLE; done<=1; end
                    else work_idx <= work_idx + 1;
                end
            end
        endcase
    end
endmodule
