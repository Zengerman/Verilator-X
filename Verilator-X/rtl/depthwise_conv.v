`timescale 1ns/1ps
module depthwise_conv #(
    parameter CH=64, FEATURE_SIZE=16, KERNEL=3
)(
    input  wire clk, reset_n, start,
    input  wire [7:0] feature_in  [0:FEATURE_SIZE-1][0:FEATURE_SIZE-1][0:CH-1],
    output reg  [7:0] feature_out [0:FEATURE_SIZE-1][0:FEATURE_SIZE-1][0:CH-1],
    input  wire        weight_we,
    input  wire [15:0] weight_addr,
    input  wire [7:0]  weight_data,
    output reg         done,
    output reg  [31:0] cycle_count
);
    localparam MAC_PER_CH = KERNEL*KERNEL;
    reg [7:0] weight_mem [0:CH-1][0:MAC_PER_CH-1];

    reg [15:0] w_ch, w_idx;
    always @(*) begin
        w_ch  = weight_addr / MAC_PER_CH;
        w_idx = weight_addr % MAC_PER_CH;
    end
    always @(posedge clk) if (weight_we && w_ch < CH)
        weight_mem[w_ch][w_idx] <= weight_data;

    localparam IDLE=2'd0, COMPUTE=2'd1, WB=2'd2;
    reg [1:0] state;
    reg [31:0] pixel, mac_iter;
    integer pi, pj, ii, jj, c, kx, ky;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state<=IDLE; done<=0; cycle_count<=0; pixel<=0; mac_iter<=0;
        end else case (state)
            IDLE: begin done<=0; if (start) begin state<=COMPUTE; pixel<=0; mac_iter<=0; end end
            COMPUTE: begin
                pi = pixel / FEATURE_SIZE;
                pj = pixel % FEATURE_SIZE;
                kx = (mac_iter / KERNEL) % KERNEL;
                ky = mac_iter % KERNEL;
                ii = pi + kx - 1;
                jj = pj + ky - 1;
                if (ii>=0 && ii<FEATURE_SIZE && jj>=0 && jj<FEATURE_SIZE)
                    for (c=0; c<CH; c=c+1)
                        feature_out[pi][pj][c] <= feature_out[pi][pj][c]
                            + feature_in[ii][jj][c] * weight_mem[c][mac_iter];
                mac_iter <= mac_iter + 1;
                cycle_count <= cycle_count + 1;
                if (mac_iter == MAC_PER_CH - 1) begin
                    mac_iter <= 0;
                    pixel <= pixel + 1;
                    if (pixel == FEATURE_SIZE*FEATURE_SIZE - 1) state <= WB;
                end
            end
            WB: begin done<=1; state<=IDLE; end
        endcase
    end
endmodule
