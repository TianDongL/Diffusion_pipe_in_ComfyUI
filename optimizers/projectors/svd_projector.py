# Adapted from galore_projector.py in galore_torch: https://github.com/jiaweizzhao/GaLore
# Added approximate SVD, SRHT projections
import torch
from .approx_svd import approximate_svd, srht, transposed_srht, get_subsample_idx_and_random_signs_from_matrix
import numpy as np


# svd decomposition
def get_orthogonal_matrix(weights, rank, proj_type, approx_svd=False, asvd_ss_scale=2):
    module_params = weights

    if module_params.data.dtype != torch.float:
        float_data = False
        original_type = module_params.data.dtype
        original_device = module_params.data.device
        matrix = module_params.data.float()
    else:
        original_type, original_device = NotImplementedError, NotImplementedError
        float_data = True
        matrix = module_params.data

    if approx_svd:
        U, s, Vh = approximate_svd(matrix, asvd_ss_scale * rank)
    else:
        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

    # make the smaller matrix always to be orthogonal matrix
    if proj_type == 'right':
        U[:, :rank] @ torch.diag(s[:rank])
        B = Vh[:rank, :]

        if not float_data:
            B = B.to(original_device).type(original_type)
        return B
    elif proj_type == 'left':
        A = U[:, :rank]
        torch.diag(s[:rank]) @ Vh[:rank, :]
        if not float_data:
            A = A.to(original_device).type(original_type)
        return A
    elif proj_type == 'full':
        A = U[:, :rank]
        B = Vh[:rank, :]
        if not float_data:
            A = A.to(original_device).type(original_type)
            B = B.to(original_device).type(original_type)
        return [A, B]
    else:
        raise ValueError('type should be left, right or full')


class SVDProjector:
    """
    This should be created for every parameter
    """
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std', approx_svd=False,
                 asvd_rank_scale=2, param_shape=None, srht_mem_efficient=False):
        super().__init__()
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

        self.approx_svd = approx_svd
        self.asvd_rank_scale = asvd_rank_scale

        self.srht_seed = None
        self.srht_states = None
        self.srht_store_proj_matrix = not srht_mem_efficient
        self.param_shape = param_shape

    def project(self, full_rank_grad: torch.Tensor, n_iter):
        if self.proj_type == 'srht':  # here we do not even compute the svd anywhere
            if full_rank_grad.shape[1] > full_rank_grad.shape[0]:
                if n_iter is not None and (self.srht_states is None or n_iter % self.update_proj_gap == 0):
                    # update the SRHT matrix
                    self.param_shape = full_rank_grad.shape
                    self.srht_seed = np.random.randint(1000000000)
                    self.srht_states = get_subsample_idx_and_random_signs_from_matrix(
                        full_rank_grad, self.rank, self.srht_seed)
                low_rank_grad = srht(full_rank_grad, subsample_size=self.rank, states=self.srht_states)
            else:
                if n_iter is not None and (self.srht_states is None or n_iter % self.update_proj_gap == 0):
                    # update the SRHT matrix
                    self.param_shape = full_rank_grad.shape
                    self.srht_seed = np.random.randint(1000000000)
                    self.srht_states = get_subsample_idx_and_random_signs_from_matrix(
                        full_rank_grad.T, self.rank, self.srht_seed)
                low_rank_grad = srht(full_rank_grad.T, subsample_size=self.rank, states=self.srht_states).t()

        elif self.proj_type == 'svd':  # this is SVD
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if n_iter is not None and (self.ortho_matrix is None or n_iter % self.update_proj_gap == 0):
                    self.ortho_matrix = get_orthogonal_matrix(
                        full_rank_grad, self.rank, proj_type='right', approx_svd=self.approx_svd,
                        asvd_ss_scale=self.asvd_rank_scale)
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if n_iter is not None and (self.ortho_matrix is None or n_iter % self.update_proj_gap == 0):
                    self.ortho_matrix = get_orthogonal_matrix(
                        full_rank_grad, self.rank, proj_type='left', approx_svd=self.approx_svd,
                        asvd_ss_scale=self.asvd_rank_scale)
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'reverse_svd':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or n_iter % self.update_proj_gap == 0:
                    self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, proj_type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or n_iter % self.update_proj_gap == 0:
                    self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, proj_type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or n_iter % self.update_proj_gap == 0:
                self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, proj_type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or n_iter % self.update_proj_gap == 0:
                self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, proj_type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or n_iter % self.update_proj_gap == 0:
                self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, proj_type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
        else:
            raise NotImplementedError("should not be here")

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'srht':
            if low_rank_grad.shape[0] < low_rank_grad.shape[1]:
                full_rank_grad = transposed_srht(low_rank_grad, original_size=self.param_shape[0],
                                                 states=self.srht_states)
            else:
                full_rank_grad = transposed_srht(low_rank_grad.T, original_size=self.param_shape[1],
                                                 states=self.srht_states).t()
        elif self.proj_type == 'svd':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_svd':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
        else:
            raise NotImplementedError("should not be here")

        return full_rank_grad * self.scale

    def to(self, device):
        if self.ortho_matrix is not None:
            self.ortho_matrix = self.ortho_matrix.to(device)
        if self.srht_states is not None:
            for k, v in self.srht_states.items():
                if torch.is_tensor(v):
                    self.srht_states[k] = v.to(device)
