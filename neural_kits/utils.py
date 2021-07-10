import random

from torch.optim.optimizer import Optimizer
import torch
import numpy as np
import torch.nn.functional as F

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers.
    Args:
        random_seed: Desired random seed.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def batch_iter(X, *tensors, batch_size=256):
    r"""Creates iterator over tensors.
    Args:
        X (torch.tensor): Feature tensor (shape: num_instances x num_features).
        tensors (torch.tensor): Target tensors (shape: num_instances).
        batch_size (int, Optional): Batch size. (default: :obj:`256`)
    """
    idxs = torch.randperm(X.size(0))
    if X.is_cuda:
         idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        res = [X[batch_idxs]]
        for tensor in tensors:
            res.append(tensor[batch_idxs])
        yield res

def diagu_indices(n, k_min=1, k_max=None):
    if k_max is None:
        return np.array(np.triu_indices(n, 1)).T
    else:
        all_pairs = set(zip(*map(np.ndarray.tolist, np.triu_indices(n, k_min))))
        rm_pairs = set(zip(*map(np.ndarray.tolist, np.triu_indices(n, 1 + k_max))))
        pairs = all_pairs - rm_pairs
        return np.array(list(pairs))


def onlywithin_indices(sequence_lengths, k_min=1, k_max=None):
    cum_n = 0
    pair_arrays = []
    for i, n in enumerate(sequence_lengths):
        pairs = diagu_indices(n, k_min, k_max) + cum_n
        pair_arrays.append(np.hstack([np.ones((pairs.shape[0], 1))*i, pairs]))
        cum_n += n
    return np.concatenate(pair_arrays).astype(int)


class MetricLogger:
    r"""Keeps track of training and validation curves, by recording:
        - Last value of train and validation metrics.
        - Train and validation metrics corresponding to maximum or minimum validation metric value.
        - Exponential moving average of train and validation metrics.
    Args:
        smoothing_factor (float, Optional): Smoothing factor used in exponential moving average.
            (default: :obj:`0.4`).
        max (bool, Optional): If :obj:`True`, tracks max value. Otherwise, tracks min value. (default: :obj:`True`).
    """
    def __init__(self, smoothing_factor=0.4, max=True):
        self.smoothing_factor = smoothing_factor
        self.max = max

        # init variables
        # last
        self.train_last = None
        self.val_last = None
        self.test_last = None

        # moving average
        self.train_smooth = None
        self.val_smooth = None
        self.test_smooth = None

        # max
        self.train_minmax = None
        self.val_minmax = None
        self.test_minmax = None
        self.step_minmax = None

    def __repr__(self):
        out = "Last: (Train) %.4f (Val) %.4f\n" % (self.train_last, self.val_last)
        out += "Smooth: (Train) %.4f (Val) %.4f\n" % (self.train_smooth, self.val_smooth)
        out += "Max: (Train) %.4f (Val) %.4f\n" % (self.train_minmax, self.val_minmax)
        return out

    def update(self, train_value, val_value, test_value=0., step=None):
        # last values
        self.train_last = train_value
        self.val_last = val_value
        self.test_last = test_value

        # exponential moving average
        self.train_smooth = self.smoothing_factor * train_value + (1 - self.smoothing_factor) * self.train_smooth \
            if self.train_smooth is not None else train_value
        self.val_smooth = self.smoothing_factor * val_value + (1 - self.smoothing_factor) * self.val_smooth \
            if self.val_smooth is not None else val_value
        self.test_smooth = self.smoothing_factor * test_value + (1 - self.smoothing_factor) * self.test_smooth \
            if self.test_smooth is not None else test_value

        # max/min validation accuracy
        if self.val_minmax is None or (self.max and self.val_minmax < val_value) or \
                (not self.max and self.val_minmax > val_value):
            self.train_minmax = train_value
            self.val_minmax = val_value
            self.test_minmax = test_value
            if step:
                self.step_minmax = step

    def __getattr__(self, item):
        if item not in ['train_min', 'train_max', 'val_min', 'val_max', 'test_min', 'test_max']:
            raise AttributeError
        if self.max and item in ['train_min', 'val_min', 'test_min']:
            raise AttributeError('Tracking maximum values, not minimum.')
        if not self.max and item in ['train_max', 'val_max', 'test_max']:
            raise AttributeError('Tracking minimum values, not maximum.')

        if 'train' in item:
            return self.train_minmax
        elif 'val' in item:
            return self.val_minmax
        elif 'test' in item:
            return self.test_minmax


class CosineLoss(torch.nn.Module):
    r"""Cosine loss.
    .. note::
        Also known as normalized L2 distance.
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        outputs = F.normalize(outputs, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)
        return (2 - 2 * (outputs * targets).sum(dim=-1)).mean()


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0) ("\beta")
        dampening (float, optional): dampening for momentum (default: 0)
        eta (float, optional): LARS coefficient
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks: https://arxiv.org/abs/1708.03888
    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, eta=0.001, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov, eta=eta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        r"""Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            eta = group['eta']
            nesterov = group['nesterov']
            lr = group['lr']
            lars_exclude = group.get('lars_exclude', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if lars_exclude:
                    local_lr = 1.
                else:
                    weight_norm = torch.norm(p).item()
                    grad_norm = torch.norm(d_p).item()
                    # Compute local learning rate for this layer
                    local_lr = eta * weight_norm / \
                        (grad_norm + weight_decay * weight_norm)

                actual_lr = local_lr * lr
                d_p = d_p.add(p, alpha=weight_decay).mul(actual_lr)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                                torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(-d_p)

        return loss