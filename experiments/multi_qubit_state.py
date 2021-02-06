from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

import git.Projektinis.tools.simulators as tools
import git.Projektinis.tools.functions as fun
import git.Projektinis.tools.gates as my_gt


def quiskit_simuliation():
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")
    qc = QuantumCircuit(qr, cr)

    qc.h(qr[0])
    qc.cx(qr[0],qr[1])

    # qc.x(qr[0])
    # qc.x(qr[1])

    qc.barrier()

    qc.measure(qr[0], cr[0])  # measure q[1] -> c[0];
    qc.measure(qr[1], cr[1])  # measure q[2] -> c[1];

    tools.simulate_and_show_result(qc)
    # tools.simulate_bloch_sphere(qc, "Paulio - Y")


quiskit_simuliation()

def math_simuliation():
    h_op_i = fun.tensor_mul(my_gt.I.get_value(),my_gt.H.get_value())
    # fun.to_latex(h_op_i)
    cx = my_gt.CXq0q1.get_value()
    cx_x_h = fun.mul(cx,h_op_i)
    # fun.to_latex(cx_x_h)
    zero = my_gt.get_zero_ket(4)
    r = fun.mul(cx_x_h,zero)
    fun.to_latex(r)
    fun.printProb(r)

# math_simuliation()

