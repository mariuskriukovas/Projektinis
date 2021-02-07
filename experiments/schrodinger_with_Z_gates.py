import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
import seaborn as sns

import git.Projektinis.tools.functions as fun
import git.Projektinis.tools.gates as my_gt
import git.Projektinis.tools.simulators as tools


def schrodinger_circle_with_Z_gates(fi, measure = True):
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")

    qc = QuantumCircuit(qr, cr)

    qc.p(np.pi, qr[0])  # u1(pi) q[1];
    qc.h(qr[1])  # h q[2];
    qc.barrier()

    qc.cx(qr[1],qr[0]) # cx q[2],q[1]
    qc.barrier()

    qc.p(np.pi, qr[0])  # u1(pi) q[1];
    qc.u(np.pi/2, np.pi/4,np.pi, qr[1]) # u2(pi/4,pi) q[2];
    qc.barrier()

    qc.cx(qr[1],qr[0])  # cx q[2],q[1];
    qc.barrier()

    qc.p(-np.pi/4, qr[0])  # u1(-pi/4) q[1];
    qc.barrier()

    qc.cx(qr[1], qr[0])  # cx q[2],q[1];
    qc.barrier()

    qc.p(np.pi/4+ (fi), qr[0]) # u1(pi/4+ ( fi* ) ) q[1];
    qc.u(np.pi/2, 2*fi,0, qr[1])# u2(2* ( fi* ) ,0) q[2];
    qc.barrier()

    qc.z(qr[0])  # cx q[2],q[0];
    qc.z(qr[1])  # cx q[2],q[0];
    qc.barrier()

    qc.u(np.pi/2, 3*np.pi/4, np.pi, qr[1]) # u2(3*pi/4,pi) q[2];
    qc.barrier()

    qc.cx(qr[1], qr[0])# cx q[2],q[1];
    qc.barrier()
    qc.p(np.pi/4, qr[0])# u1(pi/4) q[1];
    qc.barrier()
    qc.cx(qr[1], qr[0])# cx q[2],q[1];
    qc.barrier()

    qc.u(np.pi/2, 0,3 * np.pi / 4, qr[0])# u2(0,3*pi/4) q[1];
    qc.barrier()

    if measure:
        qc.measure(qr[0], cr[0]) # measure q[1] -> c[0];
        qc.measure(qr[1], cr[1])# measure q[2] -> c[1];

    return qc

def count_state_averge(val):
    val = np.abs(val)
    print(val)
    values = tools.all_binary_combs(size=2)
    s = (4, 4)
    init_data = np.zeros(s)
    df = pd.DataFrame(data=val, index=values[::-1], columns=values)
    sns.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True).set_title(
        "Šredingerio lygties su Z keitiniu perėjimo matricos reikšmių moduliai")
    plt.show()

def schrodinger_with_Z_final_state_average():
    pi_val = np.pi / 2

    qc = schrodinger_circle_with_Z_gates(pi_val, measure=False)
    res = tools.simulate_unitary_matrix_df(qc)
    count_state_averge(res)

def schrodinger_with_Z_gates_experiment():
    pi_val = np.pi/2
    qc = schrodinger_circle_with_Z_gates(pi_val)
    qc.draw(output='mpl')
    # plt.show()
    counts = tools.simulate(qc).get_counts()
    print(counts)
    plot_histogram(counts, title="Šredingerio lygties būsenos, su Z vartų keitiniu, kai $\Phi = \\frac{\pi}{2}$ ")
    plt.show()

schrodinger_with_Z_final_state_average()
# schrodinger_with_Z_gates_experiment()


#########################################################
#UNITARY
def schrodinger_circle_with_Z_gates_unitary(fi):
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")

    qc = QuantumCircuit(qr, cr)

    qc.p(np.pi, qr[0])  # u1(pi) q[1];
    qc.h(qr[1])  # h q[2];
    qc.barrier()

    qc.cx(qr[1],qr[0]) # cx q[2],q[1]
    qc.barrier()

    return qc


def schrodinger_with_Z_gates_unitary_experiment():
    pi_val = np.pi/2

    qc = schrodinger_circle_with_Z_gates_unitary(pi_val)
    qc.draw(output='mpl')
    plt.show()

    result = tools.simulate_unitary(qc)
    unitary = result.get_unitary(qc, decimals=3)
    fun.to_latex(unitary)
    return unitary

def cx_gate():
    ket_zero = np.array([[1], [0]])
    bra_zero = fun.calc_bra(ket_zero)
    bra_ket_zero = fun.mul(ket_zero, bra_zero)

    ket_one = np.array([[0], [1]])
    bra_one = fun.calc_bra(ket_one)
    bra_ket_one = fun.mul(ket_one, bra_one)

    tensor = fun.tensor_all([bra_ket_zero, my_gt.I.get_value(), ])
    tensor_2 = fun.tensor_all([bra_ket_one, my_gt.X.get_value(), ])

    cx = tensor + tensor_2
    return cx

def math_proof():
    # H (*) P(pi)
    tensor = fun.tensor_all([my_gt.H.get_value(), my_gt.p_gate.get_value(np.pi)])

    cx = cx_gate()
    res = fun.mul(cx, tensor)

    fun.to_latex(res)

    return res


# simulated_result = schrodinger_with_Z_gates_unitary_experiment()
# counted_result = math_proof()
# print("Ar lygu su pakalaida e", fun.compare_vec(simulated_result, counted_result))


def schrodinger_with_Z_gates_unitary_full_experiment():
    pi_val = np.pi/2

    qc = schrodinger_circle_with_Z_gates(pi_val, measure = False)
    qc.draw(output='mpl')
    plt.show()

    result = tools.simulate_unitary(qc)
    unitary = result.get_unitary(qc, decimals=3)

    ket_zero = np.array([[1], [0],[0], [0]])
    res = fun.mul(unitary, ket_zero)
    fun.to_latex(res)
    return res

# schrodinger_with_Z_gates_unitary_full_experiment()