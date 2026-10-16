from __future__ import annotations


from guidance.targets_pi05_robocasa import ActionSpec
from guidance.targets_pi05_libero import compute_agf_adjoint_targets_libero
from training.smolvla import score_action


def agf_targets_libero(policy, critic, guidance_model, batch, args, first_batch: bool = False):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    spec = ActionSpec(args.critic_horizon, args.action_dim)
    return compute_agf_adjoint_targets_libero(
        policy=policy,
        critic=critic,
        guidance_model=guidance_model,
        batch=batch,
        score_action_fn=score_action,
        make_att_2d_masks_fn=make_att_2d_masks,
        spec=spec,
        critic_image_size=args.critic_image_size,
        num_steps=args.agf_num_steps,
        guidance_weight=1.0,
        use_guidance_rollout=(guidance_model is not None and not first_batch),
    )
