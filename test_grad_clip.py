"""Global grad clipping uses the constructor eps, not the last group."""
import torch

from adan import Adan


def _scale(groups, foreach):
    params = []
    opt_groups = []
    for eps, grad in groups:
        param = torch.nn.Parameter(torch.zeros(4))
        param.grad = grad.clone()
        params.append(param)
        opt_groups.append({"params": [param], "eps": eps})
    opt = Adan(opt_groups, lr=1e-3, max_grad_norm=1.0,
               foreach=foreach, weight_decay=0.0)
    opt.step()
    return params[0].grad[0].item()


def test_last_group_eps_does_not_set_the_clip():
    ones = torch.ones(4)  # norm 2
    zeros = torch.zeros(4)
    # constructor eps is 1e-8. The second group overrides eps for its
    # own denominator only.
    got = _scale([(1e-8, ones), (1.0, zeros)], foreach=False)
    assert abs(got - 1.0 / (2.0 + 1e-8)) < 1e-6
    assert abs(got - 1.0 / 3.0) > 1e-3


def test_foreach_matches_single():
    ones = torch.ones(4)
    zeros = torch.zeros(4)
    groups = [(1e-8, ones), (1.0, zeros)]
    assert abs(_scale(groups, False) - _scale(groups, True)) < 1e-6


if __name__ == "__main__":
    test_last_group_eps_does_not_set_the_clip()
    test_foreach_matches_single()
    print("ok")
