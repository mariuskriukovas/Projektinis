import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

import git.Projektinis.tools.functions as fun
import git.Projektinis.tools.simulators as tools

pi_values = ['0', 'pi/6', 'pi/4', 'pi/3', 'pi/2', '(2*pi)/3', '(3*pi)/4', '(5*pi)/6', 'pi', '(7*pi)/6', '(5*pi)/4',
             '(4*pi)/3', '(3*pi)/2', '(5*pi)/3', '(7*pi)/4', '(11*pi)/6', '(9*pi)/4', ]
pi_arr = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, (2 * np.pi) / 3, (3 * np.pi) / 4, (5 * np.pi) / 6, np.pi,
          (7 * np.pi) / 6, (5 * np.pi) / 4, (4 * np.pi) / 3, (3 * np.pi) / 2, (5 * np.pi) / 3, (7 * np.pi) / 4,
          (11 * np.pi) / 6, (9 * np.pi) / 4]


def schrodinger_circle(fi, measure=True):
    qr = QuantumRegister(3, "qr")
    cr = ClassicalRegister(2, "cr")

    qc = QuantumCircuit(qr, cr)

    qc.p(np.pi, qr[1])  # u1(pi) q[1];
    qc.h(qr[2])  # h q[2];

    qc.cx(qr[2], qr[1])  # cx q[2],q[1]

    qc.p(np.pi, qr[1])  # u1(pi) q[1];
    qc.u(np.pi / 2, np.pi / 4, np.pi, qr[2])  # u2(pi/4,pi) q[2];

    qc.cx(qr[2], qr[1])  # cx q[2],q[1];

    qc.p(-np.pi / 4, qr[1])  # u1(-pi/4) q[1];

    qc.cx(qr[2], qr[1])  # cx q[2],q[1];

    qc.p(np.pi / 4 + (fi), qr[1])  # u1(pi/4+ ( fi* ) ) q[1];
    qc.u(np.pi / 2, 2 * fi, 0, qr[2])  # u2(2* ( fi* ) ,0) q[2];

    qc.cx(qr[2], qr[0])  # cx q[2],q[0];
    qc.cx(qr[1], qr[0])  # cx q[1],q[0];

    qc.p(2 * fi, qr[0])  # u1( 2*( fi* ) ) q[0];

    qc.cx(qr[1], qr[0])  # cx q[1],q[0];
    qc.cx(qr[2], qr[0])  # cx q[2],q[0];

    qc.u(np.pi / 2, 3 * np.pi / 4, np.pi, qr[2])  # u2(3*pi/4,pi) q[2];

    qc.cx(qr[2], qr[1])  # cx q[2],q[1];
    qc.p(np.pi / 4, qr[1])  # u1(pi/4) q[1];
    qc.cx(qr[2], qr[1])  # cx q[2],q[1];

    qc.u(np.pi / 2, 0, 3 * np.pi / 4, qr[1])  # u2(0,3*pi/4) q[1];
    #
    # # # qc.measure(qr[0], cr[0]) # measure q[1] -> c[0];
    # # # qc.measure(qr[1], cr[1])# measure q[2] -> c[1];
    # # # qc.measure(qr[2], cr[2])  # measure q[2] -> c[1];
    # #

    if measure:
        qc.measure(qr[1], cr[0])  # measure q[1] -> c[0];
        qc.measure(qr[2], cr[1])  # measure q[2] -> c[1];

    return qc


def generate_cirques(measure=True):
    qc = []
    for pi in pi_arr:
        qc.append(schrodinger_circle(pi, measure))
    return qc


def simulate_all(qc_arr):
    columns = ['00', '01', '10', '11']

    index = list(range(len(pi_arr)))
    df = pd.DataFrame(index=index, columns=columns)
    for i in range(len(df)):
        df.iloc[i] = 0.0

    for i in range(len(pi_arr)):
        test_result = tools.simulate(qc_arr[i])
        for test in test_result.get_counts():
            value = test_result.get_counts()[test]
            df.iloc[i][test] = value
    df = df / 5000
    return df


def represent_by_pi(data):
    print(data)

    new_data = pd.DataFrame(index=pi_arr, columns=['00', '01', '10', '11'])
    for i in range(len(data.index) - 1):
        new_data.iloc[i] = data.iloc[i]

    data = new_data
    my_colors = ['green', 'blue', 'orange', 'red', ]
    pic = data.plot(title="Išmatuotų būsenų priklausomybė  ", kind='line', lw=1, fontsize=6,
                    color=my_colors,
                    use_index=True,
                    )

    plt.ylabel("Tikimybė $|\Psi (x)|^2$")
    plt.xlabel("$\Phi \in [0, 2\pi]$ ")

    # tools.add_point(plt, 0.5, 0.5, 0)

    plt.show()


def full_rotation():
    qc_arr = generate_cirques()
    simulated_data = simulate_all(qc_arr)
    print(simulated_data)
    represent_by_pi(simulated_data)


def draw_scheme(idx):
    qc_arr = generate_cirques()
    qc = qc_arr[idx]
    qc.draw(output='mpl')
    plt.show()


def print_gates(idx):
    qc_arr = generate_cirques(measure=False)
    qc = qc_arr[idx]

    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    result = job.result()

    # # print(result.ge)
    # zero = np.array([[1]])
    # for i in range(0, 7):
    #     zero = np.append(zero, [0])
    #
    answer = result.get_unitary(qc, decimals=3)
    print(answer.shape)
    answer = fun.roundVec(answer, 3)
    answer = fun.to_latex(answer)
    #
    # answer = fun.mul(answer,zero)
    # bra = fun.calc_bra(answer)
    # # print("begin = ", zero)
    # # print("final = ", answer)
    # # print("bra = ", bra)
    # # print("amplitude = ", fun.mul(bra,answer))
    # fun.printProb(answer)
    # # # print(answer.shape)
    # # fun.to_latex(answer)
    # # # print(result.get_unitary(qc, decimals=3))

    return answer


def schrodinger_experiment():
    pi_val = np.pi / 2

    qc = schrodinger_circle(pi_val)
    qc.draw(output='mpl')
    # plt.show()

    counts = tools.simulate(qc).get_counts()
    print(counts)
    plot_histogram(counts, title="Gera")
    plt.show()

full_rotation()
# schrodinger_experiment()
