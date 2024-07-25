import itertools
import torch
import deepquantum as dq

def get_subsets(n):
    """Get powerset of [0, 1, ... , n-1]"""
    subsets = [ ]
    for k in range(n+1):
        subset = [ ]
        for i in itertools.combinations(range(n), k):
            subset.append(list(i))
        subsets.append(subset)
    return subsets

def det_test(keep_idx, a, nmodes, cholesky_mat_dic2):
    """
    return the determinant of the hermitian positive-definite matrix
    """
    if len(keep_idx)==nmodes:
        print('full case, no recursive')
        mat_l = torch.cholesky(a)
    else:
        mat_l = recur_cholesky(keep_idx, a, nmodes, cholesky_mat_dic2)
        print('using recursive Cholesky mat')

    mat_det = torch.prod(mat_l.diagonal()**2)

    return mat_det, mat_l

def recur_cholesky(keep_idx, new_mat, nmodes,  cholesky_mat_dic2): #not surpporting batch input yet
    all_modes = torch.arange(nmodes)
    mask = torch.isin(all_modes, keep_idx, invert=True)
    mask = mask.to(torch.int64)
    print(mask)
    delete_modes = torch.repeat_interleave(all_modes, mask, dim=0) #取补集
    print(keep_idx, delete_modes)
    delete_modes = list(delete_modes)
    delete_modes_pre = delete_modes[:-1]
    keep_idx_pre = set(range(nmodes))-set(delete_modes_pre)
    keep_idx_pre = list(keep_idx_pre)
    idx = (delete_modes[-1] - len(delete_modes_pre)) * 2
    dim = (len(keep_idx) + 1) * 2
    dim_ = torch.arange(dim)
    print(idx, dim)
    test_1 = dim_ != idx
    test_2 = dim_ != idx+1
    test_3 = test_1.to(torch.int64) & test_2.to(torch.int64) #取与运算
    print('left1', test_1)
    print('left2', test_2)
    print("section", test_3)

    pick_idx = torch.repeat_interleave(dim_, test_3, dim=0)
    print(pick_idx)


    # reuse the Cholesky result
    print('keep', keep_idx_pre)
    cholesky_pre = cholesky_mat_dic2[tuple(keep_idx_pre)]
    Ls = cholesky_pre[pick_idx][:, pick_idx]
    print('Ls', Ls)

    # len_newmat = torch.arange(len(new_mat))
    # mask_2 = len_newmat>=idx
    # mask_2 = mask_2.to(torch.int64)
    # print('mask_2', mask_2)
    # loop_range = torch.repeat_interleave(len_newmat, mask_2, dim=0)
    # XXX next part did not support vmap
    for i in range(idx, len(new_mat)):
        for j in range(idx, i):
            if j == 0:
                z = 0.0
            else:
                z = Ls[i][:j] * Ls[j][:j].conj()
                z = z.sum()
            Ls[i, j] = (new_mat[i][j] - z) / Ls[j, j]
        if i == 0:
            z = 0.0
        else:
            z = Ls[i][:i]*Ls[i][:i].conj()
            z = z.sum()
        Ls[i, i] = torch.real(torch.sqrt(new_mat[i, i] - z))
    return Ls

def _tor_helper2_test(submat, sub_gamma, keep_idx, nmodes, cholesky_mat_dic2):
    size = submat.size()[-1]
    temp = torch.eye(size, device = submat.device)-submat
    # inv_temp = torch.linalg.inv(temp)
    sub_gamma = sub_gamma.to(temp.device, temp.dtype)
    exp_term  = sub_gamma @ torch.linalg.solve(temp, sub_gamma.conj())/2
    det_test_ = det_test(keep_idx, temp + 0j, nmodes, cholesky_mat_dic2)
    det_mat_2 = det_test_[0]
    ch_mat = det_test_[1]
    return torch.exp(exp_term)/torch.sqrt(det_mat_2), ch_mat

def get_subsets(n):
    """Get powerset of [0, 1, ... , n-1]"""
    subsets = [ ]
    for k in range(n+1):
        subset = [ ]
        for i in itertools.combinations(range(n), k):
            subset.append(list(i))
        subsets.append(subset)
    return subsets
def get_submat_tor(a, z):
    """Get submat for torontonian calculation"""
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z)
    len_ = a.size()[-1]
    idx1 = z
    idx2 = idx1 + int((len_)/2)

    idx = z.new_empty(2*len(z), dtype=torch.int)
    idx[0::2] = idx1
    idx[1::2] = idx2

#     idx = torch.cat([idx1, idx2])
#     idx = torch.sort(idx)[0]
    if a.dim() == 1:
        return a[idx]
    if a.dim() == 2:
        return a[idx][:, idx]

def main():
    nmode = 15
    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=5, backend='gaussian')
    for i in range(nmode):
        cir.s(wires=[i])
        cir.d(wires=[i])
    for i in range(nmode-1):
        cir.bs(wires=[i,i+1])
    cir.to(torch.double)
    # final_state = [1]*nmode
    # final_state_double = torch.cat(final_state)
    covs, means = cir()
    cov_ladder = dq.photonic.qmath.quadrature_to_ladder(covs[0])
    mean_ladder = dq.photonic.qmath.quadrature_to_ladder(means[0])
    q = cov_ladder + torch.eye(2 * nmode) / 2
    gamma = mean_ladder.conj().mT @ torch.inverse(q)
    o_mat = torch.eye(2 * nmode) - torch.inverse(q)
    o_mat = o_mat
    gamma = gamma.squeeze()
    nmodes = int(len(o_mat)/2)
    if not isinstance(o_mat, torch.Tensor):
        o_mat = torch.tensor(o_mat)
    if gamma is None:
        gamma = torch.zeros(len(o_mat))
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.tensor(gamma)
    assert len(o_mat) % 2 == 0, 'input matrix dimension should be even '
    m = len(o_mat) // 2
    z_sets = get_subsets(m)
    tor = (-1) ** m

    for i in range(len(z_sets)-1, 13, -1):
        subset = z_sets[i]
        num_ = len(subset[0])
        sub_mats = torch.vmap(get_submat_tor, in_dims=(None, 0))(o_mat, torch.tensor(subset))
        sub_gammas = torch.vmap(get_submat_tor, in_dims=(None, 0))(gamma, torch.tensor(subset))
        vmap_result = torch.vmap(_tor_helper2_test, in_dims=(0,0,0,None,None))(sub_mats, sub_gammas, torch.tensor(subset), nmodes, cholesky_mat_dic2 )
        coeff = vmap_result[0]

        keys = list(map(tuple, subset))
    #     print(subset, vmap_result[1])
        cholesky_mat_dic2 = dict(zip(keys, vmap_result[1])) #construct cholesky dict
    #     print(cholesky_mat_dic)
        coeff_sum = (-1) ** (m - num_) * coeff.sum()
        tor = tor + coeff_sum
        return tor
