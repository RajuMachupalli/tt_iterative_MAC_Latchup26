# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0
"""
Gate-level cocotb tests (no internal FSM signals).

Run via: make GATES=yes  (see Makefile / gds gl_test job)
"""

import cocotb

from mac_reference import MacModel
from tb_common import apply_reset, mac_step, pack_reset, run_lockstep_outputs, setup_clock


@cocotb.test()
async def test_gl_training_regression(dut):
    """tt07 training vectors — output check only."""
    await setup_clock(dut)
    model = MacModel()
    model.reset(139)

    await apply_reset(dut, 139)
    sequence = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
    await run_lockstep_outputs(dut, model, sequence)

    assert int(dut.uo_out.value) == model.uo_out == 6
    assert int(dut.uio_oe.value) == 0x00


@cocotb.test()
async def test_gl_inference_accumulate(dut):
    """Inference mode: outputs track reference across several MAC cycles."""
    await setup_clock(dut)
    model = MacModel()
    model.reset(pack_reset(0, 5))

    await apply_reset(dut, pack_reset(0, 5))
    await run_lockstep_outputs(dut, model, [(2, 0), (3, 0), (4, 0), (5, 0)])
    assert int(dut.uio_oe.value) == 0xFF


@cocotb.test()
async def test_gl_capture_first_product(dut):
    """After S0, inference exposes product in result[23:16] on uio_out."""
    await setup_clock(dut)
    model = MacModel()
    model.reset(pack_reset(0, 7))

    await apply_reset(dut, pack_reset(0, 7))
    await mac_step(dut, 3, 0)
    model.step(3, 0)

    assert int(dut.uo_out.value) == model.uo_out
    assert int(dut.uio_out.value) == model.uio_out
    assert int(dut.uio_oe.value) == 0xFF
