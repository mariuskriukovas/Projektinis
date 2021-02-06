import matplotlib.pyplot as plt
import pandas as pd


def calc_psi(q):
    return 2 ** q  # 2 ^ n


def calc_u_mb(psi_cur):
    u_cur = psi_cur * psi_cur
    u_cur *= 80  # bytes
    u_cur /= 1048576  # mega bytes
    # u_cur /= 1024  # gyga bytes
    return u_cur


def calc_growth():
    qubits = []
    for i in range(0, 17):
        qubits.append(i)

    psi = []
    u = []
    for q in qubits:
        psi_cur = calc_psi(q)
        psi.append(psi_cur)
        u_cur = calc_u_mb(psi_cur)
        u.append(u_cur)

    # 'Kubitai': qubits,
    df = pd.DataFrame({'Būsenos': psi, 'Perėjimo matricos dydis (MB)': u})

    print(df)

    colors = ['red', 'blue']
    # title = 'Sistemos dydžio priklauso'
    ax = df.plot(color=colors)
    ax.set_xlabel("Kubitai")
    ax.set_ylabel("Dydžiai")

    plt.show()


calc_growth()


def calc_size_custom(qubits):
    psi_cur = calc_psi(qubits)
    u_cur = calc_u_mb(psi_cur)
    u_cur /= 1024
    u_cur /= 1024  # TERABYTES
    u_cur /= 1024  # PETABYTES
    print(u_cur)

# calc_size_custom(32)
