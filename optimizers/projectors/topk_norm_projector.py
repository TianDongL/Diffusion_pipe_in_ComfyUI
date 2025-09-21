# top k norm projector similar to GRASS: https://arxiv.org/abs/2406.17660
# this is a custom implementation so might not entirely match with GRASS' idea.
import torch


def top_k_norm_indices(tensor: torch.Tensor, dim: int, k: int):
    # Compute the norm along the specified dimension
    norms = torch.norm(tensor, dim=dim)

    # Get the top k indices of the largest norms
    top_k_indices = torch.topk(norms, k).indices

    return top_k_indices


class TopKNormProjector:
    """
    This should be created for every parameter
    """
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std', param_shape=None):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type

        self.top_indices = None
        self.reduced_dim = None
        assert param_shape is not None, "need param_shape to project back to original size"
        self.param_shape = param_shape

    def project(self, full_rank_grad: torch.Tensor, n_iter):

        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.top_indices is None or n_iter % self.update_proj_gap == 0:
                self.reduced_dim = 0
                self.top_indices = top_k_norm_indices(full_rank_grad, 1, self.rank)
            low_rank_grad = full_rank_grad[self.top_indices, :]
        else:
            if self.top_indices is None or n_iter % self.update_proj_gap == 0:
                self.reduced_dim = 1
                self.top_indices = top_k_norm_indices(full_rank_grad, 0, self.rank)
            low_rank_grad = full_rank_grad[:, self.top_indices]
        return low_rank_grad

    def project_back(self, low_rank_grad: torch.Tensor):
        full_rank_grad = torch.zeros(self.param_shape, dtype=low_rank_grad.dtype, device=low_rank_grad.device)
        if self.reduced_dim == 0:
            full_rank_grad[self.top_indices, :] = low_rank_grad
        else:
            full_rank_grad[:, self.top_indices] = low_rank_grad
        return full_rank_grad * self.scale

    def get_idxs(self):
        return self.top_indices