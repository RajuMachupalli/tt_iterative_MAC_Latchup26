<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This design implements a **32-bit multiply-accumulate (MAC)** unit aimed at tiny ML accelerators. A **7×8-bit multiplier** forms a 15-bit product each clock cycle. An internal FSM shifts partial sums and bias bytes to build a full 32-bit accumulated result.

**Pin usage**

| Signal | Role |
|--------|------|
| `ui_in[6:0]` | Activation (7-bit), latched while `rst_n` is low |
| `ui_in[7]` | Mode: `0` = inference (continuous accumulate), `1` = training (4-step bias sequence) |
| `ui_in[7:0]` | Weight byte (operand B) on each clock after reset |
| `uio_in[7:0]` | Bias byte during training; bidirectional pins are outputs in inference mode |
| `uo_out[7:0]` | `result[31:24]` |
| `uio_out[7:0]` | `result[23:16]` when in inference mode |

**Inference mode** (`ui_in[7]=0`): After reset, state 0 captures the first product into the upper half of `result`. The machine then stays in state 1, adding `{product, bias_byte, 0}` each cycle and shifting `result` left by 8 bits.

**Training mode** (`ui_in[7]=1`): Walks through four accumulation phases with different bias alignment. If bits `[14:11]` of the product are set (`out_th`), the FSM takes a shortened path (early saturation handling).

## How to test

Simulation tests are in `test/test.py` (cocotb). From the `test` directory:

```bash
make clean && make
```

1. Hold `rst_n` low and present `{mode, activation}` on `ui_in` (e.g. activation `2`, mode `0` → `ui_in = 8'h02`).
2. Release reset; drive weight on `ui_in[7:0]` and optional bias on `uio_in` each cycle.
3. Read `uo_out` and `uio_out` for the top 16 bits of the shifted accumulator.

## External hardware

No external hardware is required beyond the Tiny Tapeout carrier. A host MCU can stream weights and bias bytes on the GPIO pins.
