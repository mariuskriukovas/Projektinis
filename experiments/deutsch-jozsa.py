import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

import git.Projektinis.tools.simulators as tools


def test_oracle_1(qr, cr, qc):
    # oracule
    # ------------------------
    qc.x(qr[4])
    qc.barrier()
    # ------------------------

    qc.x(qr[0])
    qc.x(qr[1])
    qc.x(qr[2])
    # qc.barrier()


def test_oracle_1_return(qr, cr, qc):
    qc.x(qr[0])
    qc.x(qr[1])
    qc.x(qr[2])
    qc.barrier()


def test_oracle_2(qr, cr, qc):
    # oracule
    # ------------------------
    qc.x(qr[0])
    qc.x(qr[1])
    qc.barrier()

    qc.cx(qr[0], qr[4])
    qc.cx(qr[1], qr[4])
    qc.cx(qr[2], qr[4])
    qc.cx(qr[3], qr[4])
    qc.barrier()
    # ------------------------


def test_oracle_2_return(qr, cr, qc):
    # oracule
    # ------------------------
    qc.x(qr[0])
    qc.x(qr[1])
    qc.barrier()
    # ------------


def test_oracul_values():
    test_case = [test_oracle_1, test_oracle_2]

    for test in range(0, len(test_case)):

        values = tools.all_binary_combs(size=4)
        s = (16, 16)
        init_data = np.zeros(s)
        df = pd.DataFrame(data=init_data, index=values[::-1], columns=values)
        # print(df)

        for b in values:

            qr = QuantumRegister(5, "qr")
            cr = ClassicalRegister(5, "cr")
            qc = QuantumCircuit(qr, cr)

            for i in range(0, len(b)):
                if b[i] == '1':
                    qc.x(qr[3 - i])

            apply_test = test_case[test]
            apply_test(qr, cr, qc)
            # test_oracle_1(qr,cr,qc)
            # test_oracle_2(qr, cr, qc)

            qc.measure(qr[0], cr[0])
            qc.measure(qr[1], cr[1])
            qc.measure(qr[2], cr[2])
            qc.measure(qr[3], cr[3])
            qc.measure(qr[4], cr[4])

            # tools.simulate_and_show_result(qc, str(b))
            result = tools.simulate_and_return_result(qc)

            keys = list(result.keys())
            val = list(result.values())

            for i in range(0, len(keys)):
                value_cubits = keys[i][1:5]
                control_cubit = keys[i][0]
                if control_cubit == "1":
                    df[str(b)][value_cubits] = val[i] / 1024

        sns.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True).set_title("Tikimybė, kad $ q_{n+1} = 1 $"
                                                                               " pritaikius $U_{f_" + str(
            test + 1) + "}$")
        plt.xlabel('Kubitų $ q_0, q_1, ..., q_{n}$ įvestis', fontsize=10)  # x-axis label with fontsize 15
        plt.ylabel('Galimos $ q_0, q_1, ..., q_{n}$ reikšmės', fontsize=10)  # y-axis label with fontsize 15
        plt.show()


def show_oracul_schemes():
    qr = QuantumRegister(5, "qr")
    cr = ClassicalRegister(1, "cr")
    qc = QuantumCircuit(qr, cr)

    # oracule U 1
    # test_oracle_1(qr,cr,qc)

    # oracule U 2
    test_oracle_2(qr, cr, qc)

    qc.measure(qr[4], cr[0])

    tools.simulate_and_show_result(qc, title="$ U_{f_1} $")


# show_oracul_schemes()

def test_deutsch_jozsa():
    def test_oracle_1_full(qr, cr, qc):
        test_oracle_1(qr, cr, qc)
        test_oracle_1_return(qr, cr, qc)

    def test_oracle_2_full(qr, cr, qc):
        test_oracle_2(qr, cr, qc)
        test_oracle_2_return(qr, cr, qc)

    # test_oracle_1(qr,cr,qc)
    test_case = [test_oracle_1_full, test_oracle_2_full]
    for test in range(0, len(test_case)):
        qr = QuantumRegister(5, "qr")
        cr = ClassicalRegister(4, "cr")
        qc = QuantumCircuit(qr, cr)

        qc.h(qr[0])
        qc.h(qr[1])
        qc.h(qr[2])
        qc.h(qr[3])
        qc.x(qr[4])
        qc.barrier()

        qc.h(qr[4])
        qc.barrier()

        current_test = test_case[test]
        current_test(qr, cr, qc)

        qc.h(qr[0])
        qc.h(qr[1])
        qc.h(qr[2])
        qc.h(qr[3])

        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        qc.measure(qr[3], cr[3])

        tools.simulate_and_show_result(qc,
                                       title="Deutsch-Jozsos reikšminių kubitų pasiskirstymo tikimybės, kai $ U_{f_" + str(
                                           test + 1) + "} $",
                                       circle_title="Deutsch-Jozsos schema kai $ U_{f_" + str(test + 1) + "} $")


def test_deutsch_jozsa_with_addtional_h():
    qr = QuantumRegister(5, "qr")
    cr = ClassicalRegister(4, "cr")
    qc = QuantumCircuit(qr, cr)

    qc.h(qr[0])
    qc.h(qr[1])
    qc.h(qr[2])
    qc.h(qr[3])
    qc.x(qr[4])
    qc.barrier()

    qc.h(qr[4])
    qc.barrier()

    # --------------
    # qc.h(qr[4])
    # qc.barrier()

    test_oracle_1(qr, cr, qc)
    test_oracle_1_return(qr, cr, qc)

    # qc.h(qr[0])
    # qc.h(qr[1])
    # qc.h(qr[2])
    # qc.h(qr[3])

    tools.simulate_bloch_sphere(qc, "")
    # unitary = tools.simulate_unitary_matrix_df(qc)
    # zero = my_gt.get_zero_ket(32)
    # r = fun.mul(unitary,zero)
    # fun.printProb(r)

    # print(fun.mul(unitary,zero))
    # qc.measure(qr[0], cr[0])
    # qc.measure(qr[1], cr[1])
    # qc.measure(qr[2], cr[2])
    # qc.measure(qr[3], cr[3])
    #
    # test = 1
    # tools.simulate_and_show_result(qc,
    #                                title="Deutsch-Jozsos reikšminių kubitų pasiskirstymo tikimybės, kai $ U_{f_" + str(
    #                                    test + 1) + "} $ su papildomais Hadamardo vartais ",
    #                                circle_title="Deutsch-Jozsos schema kai $ U_{f_" + str(test + 1) + "} $ su papildomais Hadamardo vartais ")


# test_oracul_values()
# test_deutsch_jozsa()
test_deutsch_jozsa_with_addtional_h()
