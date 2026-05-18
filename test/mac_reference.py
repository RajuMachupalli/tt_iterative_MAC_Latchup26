# SPDX-License-Identifier: Apache-2.0
"""Cycle-accurate reference for tt_um_rajum_iterativeMAC FSM."""


def out_th(mult_out: int) -> bool:
    return bool(mult_out & 0x7800)


def temp_b(state: int, mult_out: int, uio_in: int) -> int:
    out = mult_out & 0x7FFF
    uio = uio_in & 0xFF
    if state == 0:
        return 0
    if state == 1:
        return (out << 16) | (uio << 8)
    if state == 2:
        return (out << 8) | uio
    if state == 3:
        return (uio << 24) | (out << 9)
    if state == 4:
        return (uio << 16) | ((out >> 8) << 9)
    if state == 5:
        return (out << 8) | uio
    if state == 6:
        return (uio << 24) | (out << 9)
    if state == 7:
        return (uio << 16) | ((out >> 8) << 9)
    return 0


def next_state(state: int, mode: int, mult_out: int) -> int:
    th = out_th(mult_out)
    if state == 0:
        return 1
    if mode == 0:
        return 1 if state == 1 else state
    # training
    if state == 1:
        return 5 if th else 2
    if state == 2:
        return 6 if th else 3
    if state == 3:
        return 7 if th else 4
    if state == 4:
        return 1
    if state == 5:
        return 6
    if state == 6:
        return 7
    if state == 7:
        return 1
    return (state + 1) & 7


class MacModel:
    def __init__(self):
        self.mode = 0
        self.inp_a = 0
        self.sum = 0
        self.result = 0
        self.state = 0

    def reset(self, ui_in: int):
        self.mode = (ui_in >> 7) & 1
        self.inp_a = ui_in & 0x7F
        self.sum = 0
        self.result = 0
        self.state = 0

    def mult(self, weight: int) -> int:
        return (self.inp_a * (weight & 0xFF)) & 0x7FFF

    def step(self, weight: int, uio_in: int) -> None:
        m = self.mult(weight)
        b = temp_b(self.state, m, uio_in)
        a = self.sum
        c = (a + b) & 0xFFFFFFFF
        ns = next_state(self.state, self.mode, m)

        if self.state == 0:
            self.result = (self.result & 0x0000_FFFF) | ((m & 0xFFFF) << 16)
        elif self.state in (4, 7):
            self.sum = 0
            self.result = c
        else:
            self.sum = c
            self.result = ((self.result << 8) & 0xFFFFFFFF)

        self.state = ns

    @property
    def uo_out(self) -> int:
        return (self.result >> 24) & 0xFF

    @property
    def uio_out(self) -> int:
        return (self.result >> 16) & 0xFF

    @property
    def uio_oe(self) -> int:
        return 0x00 if self.mode else 0xFF
