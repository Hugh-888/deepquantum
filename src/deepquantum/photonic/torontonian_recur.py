import torch
def recur_cholesky(L, Z, idx, mat):  # pragma: no cover
    """Returns the Cholesky factorization of a matrix using sub-matrix of prior

    Cholesky based on the new matrix and lower right quadrant.

    Algorithm from paper:
    https://arxiv.org/pdf/2109.04528.pdf

    Args:
        L (array): previous Cholesky
        Z (array): new sub-matrix indices
        idx: index of starting row/column of lower right quadrant
        mat (array): new matrix

    Returns:
        np.float64 or np.complex128: the Cholesky of matrix ``mat``
    """
#     Ls = thewalrus._hafnian.nb_ix(L, Z, Z)
    Ls = L[Z][:, Z]

    for i in range(idx, len(mat)):
        for j in range(idx, i):
            if j == 0:
                z = 0.0
            else:
                z = Ls[i][:j] * Ls[j][:j].conj()
                z = z.sum()
            Ls[i, j] = (mat[i][j] - z) / Ls[j, j]
        if i == 0:
            z = 0.0
        else:
            z = Ls[i][:i]*Ls[i][:i].conj()
            z = z.sum()
        Ls[i, i] = torch.real(torch.sqrt(mat[i, i] - z))
    return Ls

# def recur_cholesky2(L, Z, idx, mat):
def rec_tor_helper(L, modes, A, n):  # pragma: no cover
    """Returns the recursive Torontonian sub-computation of a matrix
    using numba.

    Algorithm from paper:
    https://arxiv.org/pdf/2109.04528.pdf

    Args:
        L (array): current Cholesky
        modes (array): optical mode
        A (array): a square, symmetric array of even dimensions
        n: size of the original matrix

    Returns:
        np.float64 or np.complex128: the recursive torontonian
        sub-computation of matrix ``A``
    """
    tot = 0.0
    if len(modes) == 0:
        start = 0
    else:
        start = modes[-1] + 1

    for i in range(int(start), n):
        nextModes = torch.cat([modes, torch.tensor([i])])# idx to remove
        dim = 2 * (n - len(modes))# left dim-2
        idx = (i - len(modes)) * 2 #被移除的位置

#         nm = Az.size(-1)[0]
        Z = torch.cat([torch.arange(idx), torch.arange(idx+2, dim)])
#         print(i, Z)
        Az = A[Z][:,Z]
        Ls = recur_cholesky(L, Z, idx, torch.eye(dim-2) - Az)
        det = torch.prod(Ls.diagonal()**2)
#         print('nextModes',nextModes, 'Z', Z, 'idx', idx, 'det', det)
        tot += ((-1) ** len(nextModes)) / torch.sqrt(det) + rec_tor_helper(Ls, nextModes, Az, n)

    return tot

def rec_torontonian_torch(A):  # pragma: no cover
    """Returns the Torontonian of a matrix using recursive approach.

    Algorithm from paper:
    https://arxiv.org/pdf/2109.04528.pdf

    Args:
        A (torch.tensor): a square, symmetric array of even dimensions

    Returns:
        the torontonian of matrix ``A``
    """
    n = int(A.size(-1)/2)
    Z = torch.empty(2*n, dtype=torch.int)
    Z[0::2] = torch.arange(0, n)
    Z[1::2] = torch.arange(n, 2*n)
    A = A[Z][:,Z]
    L = torch.linalg.cholesky(torch.eye(2 * n) - A)
    det = torch.prod(L.diagonal()**2)
    ini_mode = torch.tensor([], dtype=torch.int)
    return 1 / torch.sqrt(det) + rec_tor_helper(L, ini_mode, A, n)