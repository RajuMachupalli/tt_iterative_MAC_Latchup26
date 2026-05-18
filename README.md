![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# Iterative MAC — Tiny Tapeout (LatchUp 2026)

32-bit **multiply-accumulate** block with a **7×8-bit** multiplier, ported from [tt07_iterativeMAC](https://github.com/RajuMachupalli/tt07_iterativeMAC) for the LatchUp 2026 shuttle.

- [Project datasheet](docs/info.md)

## Overview

| Block | Description |
|-------|-------------|
| `multi` | 7×8 → 15-bit product (`inp_a` × weight byte) |
| `adder` | 32-bit accumulator |
| FSM | Inference (steady accumulate) or training (4-phase bias alignment) |

**Inference:** latch activation on reset, then present one weight byte per clock; read MSBs on `uo_out` / `uio_out`.

**Training:** set `ui_in[7]=1`; supply bias bytes on `uio_in` across four phases (S1–S4). Large products (`|product[14:11]`) take a shortened S5–S7 path. See [docs/info.md](docs/info.md) for the state diagram.

**Simulation:** `test/mac_reference.py` is a cycle-accurate golden model checked by cocotb against `DUT.state` and outputs.

## Setup

1. RTL lives in `src/` — top module `tt_um_rajum_iterativeMAC`.
2. Metadata in [info.yaml](info.yaml).
3. Run cocotb tests from `test/` — see [test/README.md](test/README.md).

## Resources

- [Tiny Tapeout](https://tinytapeout.com)
- [Original MAC repo](https://github.com/RajuMachupalli/tt07_iterativeMAC)
- [Local hardening guide](https://www.tinytapeout.com/guides/local-hardening/)
