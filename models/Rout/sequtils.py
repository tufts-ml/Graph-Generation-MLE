import torch
import torch.nn.functional as F

def compute_in_batches(f, calc_batch_size, *args, n=None):
    """
    Computes memory heavy function f(*args) in batches
    :param n: the total number of elements, optional if it cannot be determined as args[0].size(0)
    :param f: The function that is computed, should take only tensors as arguments and return tensor or tuple of tensors
    :param calc_batch_size: The batch size to use when computing this function
    :param args: Tensor arguments with equally sized first batch dimension
    :return: f(*args), this should be one or multiple tensors with equally sized first batch dimension
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    # Run all batches
    # all_res = [f(*batch_args) for batch_args in zip(*[torch.chunk(arg, n_batches) for arg in args])]
    # We do not use torch.chunk such that it also works for other classes that support slicing
    all_res = [f(*(arg[i * calc_batch_size:(i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    # Allow for functions that return None
    def safe_cat(chunks, dim=0):
        if chunks[0] is None:
            assert all(chunk is None for chunk in chunks)
            return None
        return torch.cat(chunks, dim)

    # Depending on whether the function returned a tuple we need to concatenate each element or only the result
    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)

class CachedLookup(object):

    def __init__(self, data):
        self.orig = data
        self.key = None
        self.current = None

    def __getitem__(self, key):
        assert not isinstance(key, slice), "CachedLookup does not support slicing, " \
                                           "you can slice the result of an index operation instead"

        assert torch.is_tensor(key)  # If tensor, idx all tensors by this tensor:

        if self.key is None:
            self.key = key
            self.current = self.orig[key]
        elif len(key) != len(self.key) or (key != self.key).any():
            self.key = key
            self.current = self.orig[key]

        return self.current

def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts

import torch
import torch.nn.functional as F


def _pad_mask(mask):
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(-1) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


def _mask_bool2byte(mask):
    assert mask.dtype == torch.uint8
    # assert (mask <= 1).all()  # Precondition, disabled for efficiency
    mask, d = _pad_mask(mask)
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def _mask_byte2long(mask):
    assert mask.dtype == torch.uint8
    mask, d = _pad_mask(mask)
    # Note this corresponds to a temporary factor 8
    # memory overhead by converting to long before summing
    # Alternatively, aggregate using for loop
    return (mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)).sum(-1)


def mask_bool2long(mask):
    assert mask.dtype == torch.uint8
    return _mask_byte2long(_mask_bool2byte(mask))


def _mask_long2byte(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.uint8).view(*mask.size()[:-1], -1)[..., :n]


def _mask_byte2bool(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0


def mask_long2bool(mask, n=None):
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)


def mask_long_scatter(mask, values, check_unset=True):
    """
    Sets values in mask in dimension -1 with arbitrary batch dimensions
    If values contains -1, nothing is set
    Note: does not work for setting multiple values at once (like normal scatter)
    """
    assert mask.size()[:-1] == values.size()
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))