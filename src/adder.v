/*
 * Copyright (c) 2024 Raju Machupalli
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module adder (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] c
);

  assign c = a + b;

endmodule
