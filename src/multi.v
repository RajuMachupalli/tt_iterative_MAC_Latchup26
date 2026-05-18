/*
 * Copyright (c) 2024 Raju Machupalli
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

// 7x8-bit signed-style multiply (unsigned operands, 15-bit product)
module multi (
    input  wire [6:0] a,
    input  wire [7:0] b,
    output wire [14:0] c
);

  assign c = a * b;

endmodule
