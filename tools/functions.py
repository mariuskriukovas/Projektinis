import array_to_latex as a2l
import numpy as np
from numpy.linalg import inv


def inverse(m):
    return inv(m)


def is_unitary(m):
    # print(m.dot(m.T.conj()))
    return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))


def is_ortogonal(a, b):
    return np.dot(a, b) == 0


def conjugate_transpose(a):
    return np.conjugate(a).transpose()


def modulo_pow(a):
    return (np.abs(a) ** 2)


def norm(v):
    # sum = 0
    # for vi in v:
    #     sum += moduloPow(vi)
    # return np.sqrt(sum)
    return np.linalg.norm(v)


def find_prob(v):
    p = []
    s = norm(v)
    # print("|psi| =",(s**2))
    for vi in v:
        si = norm(vi)
        pi = ((si ** 2) / (s ** 2))
        # print((si ** 2), "/", (s ** 2), "=", pi)
        p.append(pi)
    if np.abs(np.sum(p) - 1) > 0.000001:
        raise Exception("wrong probabilities")
    return p


def printProb(v):
    prob = find_prob(v)
    for i in range(0, len(v)):
        b = bin(i).replace("0b", "")
        print("( " + str(b) + " ) " + str(v[i]) + "---->" + str(prob[i]))


def mul(a, b):
    return np.matmul(a, b)


def tensor_mul(a, b):
    tensor = np.tensordot(a, b, 0)
    v = []
    for t in tensor:
        tBegin = t[0]
        for i in range(1, len(t)):
            tBegin = np.concatenate((tBegin, t[i]), axis=1)
        v.append(tBegin)

    answer = v[0]
    for i in range(1, len(v)):
        answer = np.concatenate((answer, v[i]), axis=0)
    return answer


def to_val(arr):
    return list(map(lambda x: x.get_value(), arr))


def tensor_all(arr):
    # arr = to_val(arr)
    r = arr[0]
    for i in range(1, len(arr)):
        r = tensor_mul(r, arr[i])

    return r


roundVec = np.vectorize(np.round)


def round_matrix():
    return roundVec


abs_vec = np.vectorize(np.abs)


def com_epsilon(x, epsilon):
    return x <= epsilon


cmpVec = np.vectorize(com_epsilon)


def compare_vec(a, b, epsilon=0.1):
    # absA = abs_vec(a)
    # absB = abs_vec(b)

    # print(roundVec(a,2))
    # print("-----------------------------------------------")
    # print(roundVec(b,2))

    diff = np.subtract(a, b)
    diff = abs_vec(diff)
    for i in diff:
        for j in i:
            if j > epsilon:
                return False

    return True


def to_latex(a):
    return a2l.to_ltx(a, frmt='{:6.2f}', arraytype='bmatrix')


def map_to_arr(v):
    return [v]


def to_ket(ket):
    return list(map(map_to_arr, ket))


def to_bra(bra):
    return map_to_arr(bra)


def calc_bra(v):
    return np.conjugate(v).transpose()


def calc_ermitian(v):
    return np.conjugate(v).transpose()
