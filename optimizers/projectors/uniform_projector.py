import torch


class UniformProjector:
    """
    This should be created for every parameter
    """
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, param_shape=None):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.opt_idxs = None
        self.scale = scale
        assert param_shape is not None, "need param_shape to project back to original size"
        self.param_shape = param_shape
        self.reduced_dim = None

    def project(self, full_rank_grad: torch.Tensor, n_iter):
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.opt_idxs is None or n_iter % self.update_proj_gap == 0:
                self.reduced_dim = 0
                self.opt_idxs = torch.randperm(full_rank_grad.shape[0])[:self.rank]
            low_rank_grad = full_rank_grad[self.opt_idxs, :]
        else:
            if self.opt_idxs is None or n_iter % self.update_proj_gap == 0:
                self.reduced_dim = 1
                self.opt_idxs = torch.randperm(full_rank_grad.shape[1])[:self.rank]
            low_rank_grad = full_rank_grad[:, self.opt_idxs]
        return low_rank_grad

    def project_back(self, low_rank_grad: torch.Tensor):
        full_rank_grad = torch.zeros(self.param_shape, dtype=low_rank_grad.dtype, device=low_rank_grad.device)
        if self.reduced_dim == 0:
            full_rank_grad[self.opt_idxs, :] = low_rank_grad
        else:
            full_rank_grad[:, self.opt_idxs] = low_rank_grad
        return full_rank_grad * self.scale

    def get_idxs(self):
        return self.opt_idxs