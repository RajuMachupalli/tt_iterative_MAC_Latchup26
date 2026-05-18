/*
 * Copyright (c) 2024 Raju Machupalli
 * SPDX-License-Identifier: Apache-2.0
 *
 * 32-bit iterative MAC with a 7x8-bit multiplier for ML inference/training.
 *
 * ui_in[6:0]  - activation (latched on reset)
 * ui_in[7]    - mode: 0 = inference (accumulate each cycle), 1 = training (4-byte bias walk)
 * ui_in[7:0]  - weight byte (per cycle, operand B of multiplier)
 * uio_in[7:0] - bias byte (training) or unused (inference outputs on uio)
 * uo_out      - result[31:24]
 * uio_out     - result[23:16] (outputs when mode=0)
 */

`default_nettype none

module tt_um_rajum_iterativeMAC (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

  reg        mode;
  reg [6:0]  inp_a;
  reg [31:0] sum, result;
  reg [2:0]  state;

  wire       out_th;
  reg [31:0] temp_b;
  wire [31:0] temp_a, temp_c;
  wire [14:0] mult_out;

  multi mult_i (
      .a(inp_a),
      .b(ui_in),
      .c(mult_out)
  );

  adder adder_i (
      .a(temp_a),
      .b(temp_b),
      .c(temp_c)
  );

  assign out_th = |mult_out[14:11];

  // Inference: drive uio as output; training: uio is input for bias bytes
  assign uio_oe  = mode ? 8'b0000_0000 : 8'b1111_1111;
  assign uio_out = mode ? 8'b0000_0000 : result[23:16];
  assign uo_out  = result[31:24];
  assign temp_a  = sum;

  always @(posedge clk) begin
    if (!rst_n) begin
      mode   <= ui_in[7];
      inp_a  <= ui_in[6:0];
      result <= 32'd0;
      sum    <= 32'd0;
      state  <= 3'd0;
    end else begin
      case (state)
        3'd0: begin
          result[31:16] <= mult_out;
          state         <= 3'd1;
        end
        3'd1: begin
          sum    <= temp_c;
          result <= result << 8;
          if (!mode)
            state <= 3'd1;
          else if (out_th)
            state <= 3'd5;
          else
            state <= 3'd2;
        end
        3'd2: begin
          sum    <= temp_c;
          result <= result << 8;
          if (out_th)
            state <= 3'd6;
          else
            state <= 3'd3;
        end
        3'd3: begin
          sum    <= temp_c;
          result <= result << 8;
          if (out_th)
            state <= 3'd7;
          else
            state <= 3'd4;
        end
        3'd4: begin
          sum    <= 32'd0;
          result <= temp_c;
          state  <= 3'd1;
        end
        3'd5: begin
          sum    <= temp_c;
          result <= result << 8;
          state  <= 3'd6;
        end
        3'd6: begin
          sum    <= temp_c;
          result <= result << 8;
          state  <= 3'd7;
        end
        3'd7: begin
          sum    <= 32'd0;
          result <= temp_c;
          state  <= 3'd1;
        end
        default: begin
          state  <= state + 3'd1;
          sum    <= temp_c;
          result <= result << 8;
        end
      endcase
    end
  end

  always @(*) begin
    case (state)
      3'd0: temp_b = 32'd0;
      3'd1: temp_b = {1'b0, mult_out, uio_in, 8'd0};
      3'd2: temp_b = {9'b0, mult_out, uio_in};
      3'd3: temp_b = {uio_in, 9'd0, mult_out};
      3'd4: temp_b = {8'd0, uio_in, 9'd0, mult_out[14:8]};
      3'd5: temp_b = {9'b0, mult_out, uio_in};
      3'd6: temp_b = {uio_in, 9'd0, mult_out};
      3'd7: temp_b = {8'd0, uio_in, 9'd0, mult_out[14:8]};
      default: temp_b = 32'd0;
    endcase
  end

  wire _unused = &{ena, 1'b0};

endmodule
