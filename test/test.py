# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from mac_reference import MacModel, out_th


def pack_reset(mode: int, activation: int) -> int:
    return ((mode & 1) << 7) | (activation & 0x7F)


def dut_state(dut) -> int:
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
    dut.ui_in.value = weight & 0xFF
    dut.uio_in.value = bias & 0xFF
    await ClockCycles(dut.clk, 1)


async def run_lockstep(dut, model: MacModel, steps: list[tuple[int, int]]):
    """Run RTL and reference in parallel; return list of (rtl_state, ref_state) per step."""
    trace = []
    for weight, bias in steps:
        model.step(weight, bias)
        await mac_step(dut, weight, bias)
        rtl_s = dut_state(dut)
        trace.append(
            {
                "state": rtl_s,
                "ref_state": model.state,
                "uo": int(dut.uo_out.value),
                "ref_uo": model.uo_out,
                "uio": int(dut.uio_out.value),
                "ref_uio": model.uio_out,
                "uio_oe": int(dut.uio_oe.value),
                "ref_uio_oe": model.uio_oe,
            }
        )
    return trace


def assert_states(trace, expected_states: list[int], msg: str):
    got = [t["state"] for t in trace]
    assert got == expected_states, f"{msg}: states {got} != {expected_states}"


@cocotb.test()
async def test_reset_latches_mode_and_activation(dut):
    await setup_clock(dut)
    await apply_reset(dut, pack_reset(1, 42))
    assert dut_state(dut) == 0
    await mac_step(dut, 1, 0)
    assert dut_state(dut) == 1
    assert int(dut.uio_oe.value) == 0x00


@cocotb.test()
async def test_inference_stays_in_state_1(dut):
    await setup_clock(dut)
    model = MacModel()
    model.reset(pack_reset(0, 5))

    await apply_reset(dut, pack_reset(0, 5))
    trace = await run_lockstep(
        dut,
        model,
        [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
    )
    assert_states(trace, [1, 1, 1, 1, 1], "inference must hold state 1")
    for t in trace:
        assert t["uio_oe"] == 0xFF
        assert t["state"] == t["ref_state"]
        assert t["uo"] == t["ref_uo"]
        assert t["uio"] == t["ref_uio"]


@cocotb.test()
async def test_training_full_cycle_no_threshold(dut):
    """Training must visit S1->S2->S3->S4->S1 when product[14:11]==0."""
    await setup_clock(dut)
    model = MacModel()
    model.reset(pack_reset(1, 3))

    await apply_reset(dut, pack_reset(1, 3))
    # weights 1..4 keep 3*weight < 2048
    steps = [(1, 0x10), (2, 0x20), (3, 0x30), (4, 0x40), (5, 0x50)]
    trace = await run_lockstep(dut, model, steps)

    assert_states(trace, [1, 2, 3, 4, 1], "full training cycle after initial S0->S1")
    for t in trace:
        assert t["state"] == t["ref_state"], f"FSM mismatch at state {t['state']}"
        assert t["uo"] == t["ref_uo"]
        assert t["uio_oe"] == 0x00


@cocotb.test()
async def test_training_second_full_cycle(dut):
    """Two complete S1->S2->S3->S4->S1 loops after initial S0 capture."""
    await setup_clock(dut)
    model = MacModel()
    model.reset(pack_reset(1, 4))

    await apply_reset(dut, pack_reset(1, 4))
    steps = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (1, 5), (2, 6), (3, 7), (4, 8)]
    trace = await run_lockstep(dut, model, steps)
    states = [t["state"] for t in trace]
    assert states[0:5] == [1, 2, 3, 4, 1], f"first training loop: {states[0:5]}"
    assert states[5:9] == [2, 3, 4, 1], f"second training loop: {states[5:9]}"


@cocotb.test()
async def test_training_early_exit_from_s1(dut):
    """out_th at S1: S1 -> S5 -> S6 -> S7 -> S1 (skip S2-S4)."""
    await setup_clock(dut)
    # 32 * 64 = 2048 -> bit 11 set
    model = MacModel()
    model.reset(pack_reset(1, 32))

    await apply_reset(dut, pack_reset(1, 32))
    model.step(0, 0)
    await mac_step(dut, 0, 0)  # S0

    weight = 64
    assert out_th(model.mult(weight))
    trace = await run_lockstep(
        dut,
        model,
        [(weight, 0x11), (weight, 0x22), (weight, 0x33), (weight, 0x44)],
    )
    assert_states(trace, [5, 6, 7, 1], "early exit from S1")


@cocotb.test()
async def test_training_early_exit_from_s2(dut):
    """out_th at S2: S2 -> S6 -> S7 -> S1."""
    await setup_clock(dut)
    model = MacModel()
    model.reset(pack_reset(1, 32))

    await apply_reset(dut, pack_reset(1, 32))
    model.step(1, 0)
    model.step(64, 0)
    await mac_step(dut, 1, 0)
    await mac_step(dut, 64, 0)
    assert dut_state(dut) == 2

    trace = await run_lockstep(dut, model, [(64, 0xAA), (64, 0xBB), (1, 0)])
    assert_states(trace, [6, 7, 1], "early exit from S2")


@cocotb.test()
async def test_training_early_exit_from_s3(dut):
    """out_th at S3: S3 -> S7 -> S1."""
    await setup_clock(dut)
    model = MacModel()
    model.reset(pack_reset(1, 32))

    await apply_reset(dut, pack_reset(1, 32))
    for w, b in [(1, 0), (2, 0), (64, 0)]:
        model.step(w, b)
        await mac_step(dut, w, b)
    assert dut_state(dut) == 3

    trace = await run_lockstep(dut, model, [(64, 0xCC), (2, 0)])
    assert_states(trace, [7, 1], "early exit from S3")


@cocotb.test()
async def test_capture_state0_product(dut):
    await setup_clock(dut)
    model = MacModel()
    model.reset(pack_reset(0, 7))

    await apply_reset(dut, pack_reset(0, 7))
    await mac_step(dut, 3, 0)  # S0: 7*3=21 -> result[31:16]

    assert dut_state(dut) == 1
    model.step(3, 0)
    assert int(dut.uo_out.value) == model.uo_out
    assert int(dut.uio_out.value) == model.uio_out


@cocotb.test()
async def test_regression_training_smoke(dut):
    """tt07 vectors: mode=1, activation=11; five cycles complete S0..S4 finalize."""
    await setup_clock(dut)
    model = MacModel()
    model.reset(139)

    await apply_reset(dut, 139)
    sequence = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
    for w, b in sequence:
        model.step(w, b)
        await mac_step(dut, w, b)
        assert dut_state(dut) == model.state

    assert dut_state(dut) == 1
    assert int(dut.uo_out.value) == model.uo_out == 6


@cocotb.test()
async def test_lockstep_random_training(dut):
    """Random small operands: RTL must match reference for 3 full training rounds."""
    await setup_clock(dut)
    model = MacModel()
    model.reset(pack_reset(1, 5))

    await apply_reset(dut, pack_reset(1, 5))
    await mac_step(dut, 0, 0)

    steps = []
    for rnd in range(3):
        for w in (1, 2, 3, 4):
            steps.append((w, (rnd * 4 + w) & 0xFF))

    trace = await run_lockstep(dut, model, steps)
    for i, t in enumerate(trace):
        assert t["state"] == t["ref_state"], f"step {i}: rtl S{t['state']} != ref S{t['ref_state']}"
        assert t["uo"] == t["ref_uo"], f"step {i}: uo_out mismatch"
        assert t["uio_oe"] == t["ref_uio_oe"]
