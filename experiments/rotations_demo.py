from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

import git.Projektinis.tools.simulators as tools


def prepare_scheme():
    qr = QuantumRegister(1, "qr")
    cr = ClassicalRegister(1, "cr")
    qc = QuantumCircuit(qr, cr)
    return (qr, cr, qc)


def zero_rotation():
    (qr, cr, qc) = prepare_scheme()

    # qc.measure(qr[0], cr[0])  # measure q[1] -> c[0];
    tools.simulate_bloch_sphere(qc, "|0>")


def h_rotation():
    (qr, cr, qc) = prepare_scheme()

    qc.h(qr[0])
    # qc.measure(qr[0], cr[0])  # measure q[1] -> c[0];
    tools.simulate_bloch_sphere(qc, "Hadamardo - H")


def x_rotation():
    (qr, cr, qc) = prepare_scheme()

    qc.x(qr[0])
    # qc.measure(qr[0], cr[0])  # measure q[1] -> c[0];
    tools.simulate_bloch_sphere(qc, "Paulio - X")


def z_rotation():
    (qr, cr, qc) = prepare_scheme()

    qc.h(qr[0])
    qc.z(qr[0])

    # qc.measure(qr[0], cr[0])  # measure q[1] -> c[0];
    #
    # tools.simulate_and_show_result(qc)
    tools.simulate_bloch_sphere(qc, "H + Paulio - Z")


def y_rotation():
    (qr, cr, qc) = prepare_scheme()

    qc.y(qr[0])
    # qc.z(qr[0])
    # qc.measure(qr[0], cr[0])  # measure q[1] -> c[0];
    # tools.simulate_and_show_result(qc)
    tools.simulate_bloch_sphere(qc, "Paulio - Y")


# zero_rotation()
h_rotation()
# x_rotation()
# z_rotation()
# y_rotation()

# fun.printProb(fun.mul(my_gt.rx_gate.get_value(np.pi/2), my_gt.get_zero_ket(2)))
# print(fun.mul(my_gt.rx_gate.get_value(np.pi/2), my_gt.get_zero_ket(2)))
# fun.printProb(fun.mul(my_gt.Y.get_value(), my_gt.get_zero_ket(2)))
# print(fun.mul(my_gt.rx_gate.get_value(np.pi/2), my_gt.get_zero_ket(2)))
