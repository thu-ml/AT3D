import kornia
import torch
import torch.optim

from AT3D.attacks.base import ConstrainedMethod
from AT3D.utils import input_diversity_func


class MIM(ConstrainedMethod):
    def __init__(self, model, goal, distance_metric, eps=3, iters=200, mu=1.0):
        super(MIM, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
        self.mu = mu

    # xs: attacker ys: victim
    def batch_attack_2d(
        self, pred_mask, xs, ys, ys_feat, xs_align_mats, align_size, eot_ens=0
    ):
        ys_adv = ys.clone().detach().requires_grad_(True)
        g = torch.zeros_like(ys_adv)
        earlystop = 0
        min_loss = 1000
        if eot_ens:
            xs_align_mats = xs_align_mats.repeat_interleave(eot_ens, 0)
            ys_feat = ys_feat.repeat_interleave(eot_ens, 0)
        for _ in range(self.iters):
            xs_adv = torch.clamp(ys_adv * pred_mask + xs * (1 - pred_mask), 0, 255)
            xs_advs = []
            for _ in range(eot_ens):
                xs_adv_di = input_diversity_func(xs_adv)
                xs_advs.append(xs_adv_di)
            if eot_ens:
                xs_advs = torch.cat(xs_advs)
            else:
                xs_advs = xs_adv

            xs_adv_aligns = kornia.warp_affine(xs_advs, xs_align_mats, align_size)
            features = self.model.forward(xs_adv_aligns)
            loss = self.getLoss(features, ys_feat)
            loss.backward()
            grad = ys_adv.grad
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)

            g = g * self.mu + grad

            self.model.zero_grad()
            ys_adv = self.step2d(
                ys_adv, 1.5 * self.eps / self.iters, g, xs, self.eps, pred_mask
            )
            ys_adv = ys_adv.detach().requires_grad_(True)

            if min_loss > loss.item():
                min_loss = loss.item()
                earlystop = 0
            else:
                earlystop += 1
            if earlystop >= 20:
                break

        return xs_adv.detach()

    def batch_attack(self, hyparam_str, target_name, mode, xs_key, ys_feat, **kwargs):
        kwargs_old = {key: kwargs[key].clone().detach() for key in xs_key}
        gs = {key: torch.zeros_like(kwargs[key]) for key in xs_key}

        for i in range(self.iters):
            for key in xs_key:
                features = self.model.forward(
                    "{}_{}_{}".format(target_name, hyparam_str, i), **kwargs
                )
                loss = self.getLoss(features, ys_feat)
                loss.backward(retain_graph=True)
                grad = kwargs[key].grad
                grad = grad / grad.abs().mean(dim=1, keepdim=True)
                gs[key] = gs[key] * self.mu + grad
                self.model.zero_grad()
                kwargs[key] = self.step(
                    kwargs[key],
                    self.eps / self.iters,
                    gs[key],
                    kwargs_old[key],
                    self.eps,
                )
                kwargs[key] = kwargs[key].detach().requires_grad_(True)

        return kwargs
