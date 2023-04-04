from AT3D.attacks.BIM_attack import BIM
from AT3D.attacks.MIM_attack import MIM


def attack_implementation(
    framework,
    hyparam_str,
    target_name,
    model,
    attack_method,
    iters,
    attack_mode,
    attack_eps,
    xs_key,
    ys_feat,
    **kwargs,
):
    """
    arguments:
      model:
      attack_method: bim, mim, fgsm, etc
      attack_mode: 'fixed': fixed eps, 'adam': eps adapted by adam, or 'smooth': add smoothing loss for mesh attack
      attack_eps: the initial eps
      xs_key: the list of adjusted parameters' key name
      ys_feat: y_target
      **kwargs: the other input parameters of the model

    return:
      xs_adv: adversarial xs parameters
    """
    if attack_method == "BIM":
        bim = BIM(
            framework,
            model,
            goal="impersonate",
            distance_metric="l2",
            eps=attack_eps,
            iters=iters,
        )
        new_kwargs, res_imgs = bim.batch_attack(
            hyparam_str, target_name, attack_mode, xs_key, ys_feat, **kwargs
        )
    elif attack_method == "MIM":
        mim = MIM(
            framework,
            model,
            goal="impersonate",
            distance_metric="l2",
            eps=attack_eps,
            iters=iters,
        )
        new_kwargs, res_imgs = mim.batch_attack(
            hyparam_str, target_name, attack_mode, xs_key, ys_feat, **kwargs
        )
    return new_kwargs, res_imgs


def attack_implementation_2d(
    model,
    attack_method,
    iters,
    eps,
    pred_mask,
    xs,
    ys,
    ys_feat,
    xs_align_mats,
    align_size,
    eot_ens=0,
    **kwargs,
):
    if attack_method == "MIM":
        mim = MIM(model, goal="impersonate", distance_metric="l2", eps=eps, iters=iters)
        res_imgs = mim.batch_attack_2d(
            pred_mask, xs, ys, ys_feat, xs_align_mats, align_size, eot_ens
        )
    else:
        raise NotImplementedError(f"{attack_method} is not implemented!")
    return res_imgs
