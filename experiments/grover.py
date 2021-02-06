import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

import git.Projektinis.tools.functions as fun
import git.Projektinis.tools.gates as my_gt
import git.Projektinis.tools.simulators as tools


def create_oracle(qr, cr, qc, position=0):
    i_gates = []
    for i in range(0, 4):
        i_gates.append(my_gt.I.get_value())

    i_gates = fun.tensor_all(i_gates)
    i_gates[position][position] = -1
    # print(i_gates)

    title = '$ U_{f = %s}$' % (str(position))
    qc.unitary(i_gates, [0, 1, 2, 3], label=title)
    qc.barrier()


def create_anti_oracle(qr, cr, qc, position=0):
    i_gates = []
    for i in range(0, 4):
        i_gates.append(my_gt.I.get_value())

    i_gates = fun.tensor_all(i_gates)
    i_gates[position][position] = -1
    # print(i_gates)
    i_gates = (-1) * i_gates

    title = '$ -1 * U_{f(%s)}$' % (str(position))
    qc.unitary(i_gates, [0, 1, 2, 3], label=title)
    qc.barrier()


def apply_h(qr, cr, qc):
    for q in qr:
        qc.h(q)


def apply_x(qr, cr, qc):
    for q in qr:
        qc.x(q)


def apply_m(qr, cr, qc):
    for q in range(0, len(qr)):
        qc.measure(qr[q], cr[q])


def oracle(qr, cr, qc):
    qc.cz(0, 3)
    qc.cz(1, 3)
    qc.cz(2, 3)
    qc.barrier()


def amplification(qr, cr, qc):
    apply_h(qr, cr, qc)
    qc.barrier()

    apply_x(qr, cr, qc)
    qc.barrier()

    # qc.mct([1, 2, 3], qr[0])
    # qc.barrier()
    create_anti_oracle(qr, cr, qc)

    apply_x(qr, cr, qc)
    qc.barrier()

    apply_h(qr, cr, qc)
    qc.barrier()


def get_registers():
    qr = QuantumRegister(4, "qr")
    cr = ClassicalRegister(4, "cr")
    qc = QuantumCircuit(qr, cr)
    return (qr, cr, qc)


def simulate_unitary(gate):
    (qr, cr, qc) = get_registers()
    gate(qr, cr, qc)
    unitary = tools.simulate_unitary_matrix_df(qc)
    # print(unitary)
    return unitary


def create_diff(qr, cr, qc, diff_state, title='H^{\oplus n}'):
    diff_title = '$ U_{\omega =  %s} $' % (title)
    qc.unitary(diff_state, [0, 1, 2, 3], label=diff_title)


def grover():
    (qr, cr, qc) = get_registers()

    apply_h(qr, cr, qc)
    qc.barrier()

    create_oracle(qr, cr, qc)
    # oracle(qr,cr,qc)
    amplification(qr, cr, qc)

    apply_m(qr, cr, qc)
    tools.simulate_and_show_result(qc, "labas")
    # print(tools.simulate_unitary_matrix_df(qc))


# grover()


def init_gates(qc):
    h_gates = []
    for i in range(0, 4):
        h_gates.append(my_gt.H.get_value())

    state = fun.tensor_all(h_gates)
    qc.unitary(state, [0, 1, 2, 3], label='$H^{\oplus n}$')
    return state


def count_s_state(prev_state):
    (a, _) = prev_state.shape
    zero = my_gt.get_zero_ket(a)

    ket_s_state = fun.mul(prev_state, zero)
    bra_s_state = fun.calc_bra(ket_s_state)

    ket_s_state = fun.to_ket(ket_s_state)
    bra_s_state = fun.to_bra(bra_s_state)

    bra_ket_s_state = fun.mul(ket_s_state, bra_s_state)
    bra_ket_s_state = 2 * bra_ket_s_state

    i_gates = []
    for i in range(0, 4):
        i_gates.append(my_gt.I.get_value())

    i_gates = fun.tensor_all(i_gates)
    bra_ket_s_state = bra_ket_s_state - i_gates

    return bra_ket_s_state


def grover_unitary(position, retrun_result=False):
    qr = QuantumRegister(4, "qr")
    cr = ClassicalRegister(4, "cr")
    qc = QuantumCircuit(qr, cr)
    state = None

    init_gates(qc)
    create_oracle(qr, cr, qc, position)

    def get_state(qr, cr, qc):
        apply_h(qr, cr, qc)
        # qc.barrier()
        # oracle(qr, cr, qc)

    state = simulate_unitary(get_state)
    diff_state = count_s_state(state)
    create_diff(qr, cr, qc, diff_state)

    create_oracle(qr, cr, qc, position)
    create_diff(qr, cr, qc, diff_state)

    qc.measure(qr[0], cr[0])
    qc.measure(qr[1], cr[1])
    qc.measure(qr[2], cr[2])
    qc.measure(qr[3], cr[3])

    if retrun_result:
        result = tools.simulate_and_return_result(qc)
        return result
    else:
        title = 'Groverio paieškos tikimybės, kai $U_{f = %s}$' % (str(position))
        circle_title = 'Groverio paieškos schema, kai $U_{f = %s}$' % (str(position))
        tools.simulate_and_show_result(qc, title=title, circle_title=circle_title)


def test_all_grover_values():
    values = tools.all_binary_combs(size=4)
    s = (16, 16)
    init_data = np.zeros(s)
    df = pd.DataFrame(data=init_data, index=values[::-1], columns=values)

    for j in range(0, 16):
        result = grover_unitary(j, retrun_result=True)
        keys = list(result.keys())
        val = list(result.values())

        for i in range(0, len(keys)):
            df[values[j]][keys[i]] = val[i] / 1024

    sns.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True).set_title(
        "Groverio paieškos tikimybės su skirtingais $U_{f}$ ")
    plt.xlabel('Orakulo $U_{f}$ reikšmės', fontsize=10)  # x-axis label with fontsize 15
    plt.ylabel('Galimos būsenos atlikus matavimą', fontsize=10)  # y-axis label with fontsize 15
    plt.show()


def grover_with_bell_state(pos=3):
    qr = QuantumRegister(4, "qr")
    cr = ClassicalRegister(4, "cr")
    qc = QuantumCircuit(qr, cr)

    def init_state(qr, cr, qc):
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])

        qc.h(qr[3])
        qc.cx(qr[3], qr[2])

    init_state(qr, cr, qc)
    create_oracle(qr, cr, qc, position=pos)

    state = simulate_unitary(init_state)
    diff_state = count_s_state(state)

    create_diff(qr, cr, qc, diff_state)

    apply_m(qr, cr, qc)

    # title = 'Groverio paieškos tikimybės, kai $U_{f = 1111}$'
    # circle_title = 'Groverio paieškos schema, kai $U_{f = 1111}$'
    # tools.simulate_and_show_result(qc, title=title, circle_title=circle_title)

    return tools.simulate_and_return_result(qc)


def test_all_grover_values_bell():
    values = tools.all_binary_combs(size=4)
    s = (16, 16)
    init_data = np.zeros(s)
    df = pd.DataFrame(data=init_data, index=values[::-1], columns=values)

    for j in range(0, 16):
        result = grover_with_bell_state(j)
        keys = list(result.keys())
        val = list(result.values())

        for i in range(0, len(keys)):
            df[values[j]][keys[i]] = val[i] / 1024

    sns.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True).set_title(
        "Groverio paieškos tikimybės Bello būsenoje su skirtingais $U_{f}$ ")
    plt.xlabel('Orakulo $U_{f}$ reikšmės', fontsize=10)  # x-axis label with fontsize 15
    plt.ylabel('Galimos būsenos atlikus matavimą', fontsize=10)  # y-axis label with fontsize 15
    plt.show()


# grover_another()
# test_all_grover_values()
# grover_unitary(1)
# grover_with_bell_state()
test_all_grover_values_bell()
