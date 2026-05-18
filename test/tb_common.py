# SPDX-License-Identifier: Apache-2.0
"""Shared cocotb helpers for RTL and gate-level MAC tests."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge, Timer

from mac_reference import MacModel


def pack_reset(mode: int, activation: int) -> int:
    return ((mode & 1) << 7) | (activation & 0x7F)


def dut_state(dut) -> int:
    """Call only after await sample_dut()."""
    return int(dut.DUT.state.value)


async def setup_clock(dut, period_ns: int = 20):
    clock = Clock(dut.clk, period_ns, unit="ns")
    cocotb.start_soon(clock.start())


async def apply_reset(dut, ui_reset: int, cycles: int = 5):
    dut.ena.value = 1
    dut.ui_in.value = ui_reset
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, cycles)
    dut.rst_n.value = 1


async def mac_step(dut, weight: int, bias: int = 0):
    """Drive operands and clock once (call await ReadOnly() before sampling)."""
    await Timer(1, unit="step")
    dut.ui_in.value = weight & 0xFF
    dut.uio_in.value = bias & 0xFF
    await RisingEdge(dut.clk)


async def run_lockstep_outputs(dut, model: MacModel, steps: list):
    for weight, bias in steps:
        model.step(weight, bias)
        await mac_step(dut, weight, bias)
        await ReadOnly()
        assert int(dut.uo_out.value) == model.uo_out
        assert int(dut.uio_oe.value) == model.uio_oe
        if model.uio_oe == 0xFF:
            assert int(dut.uio_out.value) == model.uio_out


async def run_lockstep(dut, model: MacModel, steps: list):
    trace = []
    for weight, bias in steps:
        model.step(weight, bias)
        await mac_step(dut, weight, bias)
        await ReadOnly()
        trace.append(
            {
                "state": dut_state(dut),
                "ref_state": model.state,
                "uo": int(dut.uo_out.value),
                "ref_uo": model.uo_out,
                "uio_oe": int(dut.uio_oe.value),
                "ref_uio_oe": model.uio_oe,
            }
        )
    return trace
