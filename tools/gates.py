import numpy as np


class Gate:
    def __init__(self, name, formula):
        self.name = name
        self.formula = formula

    def get_value(self, psi):
        return self.formula(psi)

    def get_adjusted_name(self, val):
        return str(self.name + "(" + str(val) + ")")

    def get_name(self):
        return self.name

    def print(self, psi=np.pi):
        print("<----------------------->")
        print(self.name)
        print(self.formula(psi))
        print("<----------------------->")


def init_rx_gate():
    def gate(psi):
        Rx = np.array([
            [np.round(np.cos(psi / 2), 4), (np.round(-np.sin(psi / 2), 4)) * (-1j)],
            [(np.round(np.sin(psi / 2), 4)) * (-1j), np.round(np.cos(psi / 2), 4), ],
        ])
        return Rx

    return Gate('Rx', gate)


def init_ry_gate():
    def gate(psi):
        Ry = np.array([
            [np.round(np.cos(psi / 2), 4), np.round(-np.sin(psi / 2), 4), ],
            [np.round(np.sin(psi / 2), 4), np.round(np.cos(psi / 2), 4), ],
        ])
        return Ry

    return Gate('Ry', gate)


def init_rz_gate():
    def gate(psi):
        Rz = np.array([
            [np.round(np.exp(((psi / 2) * (-1j))), 4), 0, ],
            [0, np.round(np.exp(((psi / 2) * (1j))), 4)],
        ])
        return Rz

    return Gate("Rz", gate)


def init_p_gate():
    def gate(lamb):
        gt = np.array([
            [1, 0, ],
            [0, np.round(np.exp(((lamb) * (1j))), 4)],
        ])
        return gt

    return Gate("P", gate)


def init_u_gate():
    def gate(par):
        (teta, fi, lamb) = par

        gt = np.array([
            [np.cos(teta / 2), -1 * np.round(np.exp(((lamb) * (1j))), 4) * np.sin(teta / 2), ],
            [np.round(np.exp(((fi) * (1j))), 4) * np.sin(teta / 2),
             np.round(np.exp(((fi + lamb) * (1j))), 4) * np.cos(teta / 2)],
        ])
        return gt

    return Gate("U", gate)


rx_gate = init_rx_gate()
ry_gate = init_ry_gate()
rz_gate = init_rz_gate()

p_gate = init_p_gate()
u_gate = init_u_gate()

r_gates = [rx_gate, ry_gate, rz_gate]


class SimpleGate:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get_value(self):
        return self.value

    def get_name(self):
        return self.name

    def print(self):
        print("<----------------------->")
        print(self.name)
        print(self.value)
        print("<----------------------->")


CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])

CZ = SimpleGate('CZ', CZ)

CXq0q1 = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
])

CXq0q1 = SimpleGate('CXq0q1', CXq0q1)

CXq1q0 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

CXq1q0 = SimpleGate('CXq1q0', CXq1q0)

I = np.array([
    [1, 0],
    [0, 1],
])

I = SimpleGate('I', I)

Z = np.array(
    [[1, 0],
     [0, -1]]
)

Z = SimpleGate('Z', Z)

X = np.array(
    [[0, 1],
     [1, 0]]
)

X = SimpleGate('X', X)

H = (1 / np.sqrt(2)) * np.array(
    [[1, 1],
     [1, -1]]
)

H = SimpleGate('H', H)

Y = np.array(
    [[0, -1j],
     [1j, 0]]
)

Y = SimpleGate('Y', Y)


def get_zero_ket(n):
    zero = np.array([1])
    for i in range(1, n):
        zero = np.append(zero, [0])
    return zero


def gate_factory(name, value):
    return SimpleGate(name, value)

#
# cz = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, -1]
# ])
#
# czb = np.array([
#     [0.9, 0, 0, 0],
#     [0, 1.1, 0, 0],
#     [0, 0, 0.98, 0],
#     [0, 0, 0, -0.99]
# ])
# a = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 0]
# ])
# def machGates():
#     allGates = genAllGates()
#     # for g in allGates:
#     #     e = g[3][3]
#     #     if abs(e) - 1 < 0.1 and e < 0 :
#     #         print(roundVec(g,3))
#
# machGates()
