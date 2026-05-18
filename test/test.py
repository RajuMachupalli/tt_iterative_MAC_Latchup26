# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


def pack_ui(mode: int, activation: int, weight: int = 0) -> int:
    """ui_in[7]=mode, ui_in[6:0]=activation on reset; weight in [7:0] after reset."""
    return ((mode & 1) << 7) | (activation & 0x7F) | ((weight & 0xFF) << 0)


@cocotb.test()
async def test_multiplier_baseline(dut):
    """After reset, first active cycle loads product upper half (state 0 -> 1)."""
    clock = Clock(dut.clk, 20, unit="ns")
    cocotb.start_soon(clock.start())

    dut.ena.value = 1
    dut.uio_in.value = 0
    # mode=0, activation=3, weight ignored during reset latch
    dut.ui_in.value = pack_ui(0, 3, 0)
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    # weight 4 -> product 3*4 = 12
    dut.ui_in.value = 4
    await ClockCycles(dut.clk, 1)

    assert int(dut.uo_out.value) == 0
    assert int(dut.uio_out.value) == 0
    assert int(dut.uio_oe.value) == 0xFF


@cocotb.test()
async def test_inference_accumulate(dut):
    """Inference mode: stay in state 1 and shift accumulator each cycle."""
    clock = Clock(dut.clk, 20, unit="ns")
    cocotb.start_soon(clock.start())

    dut.ena.value = 1
    dut.uio_in.value = 0
    dut.ui_in.value = pack_ui(0, 2, 0)  # activation = 2
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    # Cycle 0 (state 0): capture 2*3=6 into result[31:16]
    dut.ui_in.value = 3
    await ClockCycles(dut.clk, 1)

    # Cycle 1 (state 1): sum += {0, 2*4, 0, 0} = 8, result <<= 8
    dut.ui_in.value = 4
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 1)

    # uio_oe all ones in inference
    assert int(dut.uio_oe.value) == 0xFF

    uo = int(dut.uo_out.value)
    uio = int(dut.uio_out.value)
    dut._log.info(f"After two MAC cycles: uo_out=0x{uo:02x} uio_out=0x{uio:02x}")
    assert uo != 0 or uio != 0, "Accumulator should be non-zero after MAC steps"


@cocotb.test()
async def test_training_sequence(dut):
    """Training mode walks states 1..4 (compatible with original smoke test)."""
    clock = Clock(dut.clk, 20, unit="ns")
    cocotb.start_soon(clock.start())

    dut.ena.value = 1
    dut.ui_in.value = 139  # mode=1, activation=11
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    sequence = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
    for weight, bias in sequence:
        dut.ui_in.value = weight
        dut.uio_in.value = bias
        await ClockCycles(dut.clk, 1)

    assert int(dut.uio_oe.value) == 0x00, "uio should be input in training mode"

    await ClockCycles(dut.clk, 1)
    final_uo = int(dut.uo_out.value)
    dut._log.info(f"Training sequence complete, uo_out={final_uo}")
    assert final_uo == 6, f"Expected uo_out==6 from regression, got {final_uo}"
