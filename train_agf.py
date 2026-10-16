from __future__ import annotations


def train_smolvla(args):
    from critics.libero import load_critic
    from guidance.networks_libero import (
        GuidanceConfig,
        GuidanceNet,
        image_encoders_from_critic,
        load_guidance_checkpoint,
        save_guidance_checkpoint,
    )
    from guidance.targets_smolvla import ActionSpec
    from training.smolvla import (
        agf_targets as compute_targets,
        make_agf_dataloader,
        metric_row,
        next_agf_batch,
        region_mse,
        save_log_and_plot,
        save_training_outputs,
        set_trainable,
    )
    import os
    import random
    import torch

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    loader = make_agf_dataloader(args)
    iterator = iter(loader)

    policy = SmolVLAPolicy.from_pretrained(args.pretrained_path).to(args.device).eval()
    critic = load_critic(args.critic_ckpt, args.device).eval()

    for module in (policy, critic):
        for p in module.parameters():
            p.requires_grad_(False)

    image_encoders = image_encoders_from_critic(critic, ensemble_size=args.ensemble_size)

    first_batch, iterator = next_agf_batch(iterator, loader, policy, args.device)
    first_target = compute_targets(policy, critic, None, first_batch, args, first_batch=True)
    spec = ActionSpec(args.critic_horizon, args.action_dim)

    cfg = GuidanceConfig(
        guidance_type="agf_discrete",
        action_dim=args.action_dim,
        critic_horizon=args.critic_horizon,
        chunk_size=first_target["x_t"].shape[1],
        max_action_dim=first_target["x_t"].shape[2],
        image_feature_dim=args.image_feature_dim,
        ensemble_size=args.ensemble_size,
        hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",")),
        layer_norm=not args.no_layer_norm,
        time_embed_dim=args.time_embed_dim,
        target_scale=args.target_scale,
    )
    model = GuidanceNet(cfg, image_encoders).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 1
    if args.resume_from:
        loaded_model, _, extra = load_guidance_checkpoint(
            args.resume_from, image_encoders, device=args.device, load_optimizer=True
        )
        model.load_state_dict(loaded_model.state_dict())
        if extra.get("optimizer_state"):
            optim.load_state_dict(extra["optimizer_state"])
        start_step = (extra.get("step") or 0) + 1
        print(f"resumed from {args.resume_from} at step {start_step - 1}")

    log = []
    final_step = start_step + args.n_steps - 1

    for step in range(start_step, final_step + 1):
        if args.overfit_first_batch:
            target_batch = first_target
        else:
            batch, iterator = next_agf_batch(iterator, loader, policy, args.device)
            target_batch = compute_targets(policy, critic, model, batch, args)

        set_trainable(model, True)

        image = target_batch["image"].to(args.device)
        image2 = target_batch["image2"].to(args.device)
        x_t = target_batch["x_t"].to(args.device)
        t = target_batch["t"].to(args.device)
        target = target_batch["adjoint_target"].to(args.device)

        pred = model(image, image2, x_t, t)
        loss = region_mse(pred, target, spec)

        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optim.step()

        last_metrics = None
        if step == start_step or step % args.log_every == 0 or step == final_step:
            adj = target_batch["adjoint_norm"].float()
            b = args.batch_size
            extra = {
                "seed_norm": float(target_batch["seed_norm"].float().mean().cpu()),
                "adjoint_norm": float(adj.mean().cpu()),
                "adjoint_norm_s0": float(adj[:b].mean().cpu()),
                "q_value": float(target_batch["q_value"].float().mean().cpu()),
                "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            }
            last_metrics = metric_row(step, loss, pred, target, spec, extra)
            log.append(last_metrics)
            save_log_and_plot(log, args.output_dir)

        if args.save_every and step % args.save_every == 0:
            if last_metrics is None:
                adj = target_batch["adjoint_norm"].float()
                b = args.batch_size
                extra = {
                    "seed_norm": float(target_batch["seed_norm"].float().mean().cpu()),
                    "adjoint_norm": float(adj.mean().cpu()),
                    "adjoint_norm_s0": float(adj[:b].mean().cpu()),
                    "q_value": float(target_batch["q_value"].float().mean().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                }
                last_metrics = metric_row(step, loss, pred, target, spec, extra)
            save_guidance_checkpoint(
                os.path.join(args.output_dir, f"ckpt_step_{step}.pt"),
                model,
                cfg,
                optimizer=optim,
                step=step,
                metrics=last_metrics,
            )

    save_training_outputs(args, model, cfg, optim, log, "agf_discrete_guidance.pt")
    return model


def train_pi05_libero(args):
    from critics.libero import load_critic
    from guidance.networks_libero import (
        GuidanceConfig,
        GuidanceNet,
        image_encoders_from_critic,
        load_guidance_checkpoint,
        save_guidance_checkpoint,
    )
    from guidance.targets_pi05_robocasa import ActionSpec
    from training.pi05_libero import agf_targets_libero as compute_targets
    from training.smolvla import (
        make_agf_dataloader,
        metric_row,
        next_agf_batch,
        region_mse,
        save_log_and_plot,
        save_training_outputs,
        set_trainable,
    )
    import os
    import random
    import torch

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    loader = make_agf_dataloader(args)
    iterator = iter(loader)

    policy = PI05Policy.from_pretrained(args.pretrained_path).to(args.device).eval()
    critic = load_critic(args.critic_ckpt, args.device).eval()

    for module in (policy, critic):
        for p in module.parameters():
            p.requires_grad_(False)

    image_encoders = image_encoders_from_critic(critic, ensemble_size=args.ensemble_size)

    first_batch, iterator = next_agf_batch(iterator, loader, policy, args.device)
    first_target = compute_targets(policy, critic, None, first_batch, args, first_batch=True)
    spec = ActionSpec(args.critic_horizon, args.action_dim)

    cfg = GuidanceConfig(
        guidance_type="agf_discrete",
        action_dim=args.action_dim,
        critic_horizon=args.critic_horizon,
        chunk_size=first_target["x_t"].shape[1],
        max_action_dim=first_target["x_t"].shape[2],
        image_feature_dim=args.image_feature_dim,
        ensemble_size=args.ensemble_size,
        hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",")),
        layer_norm=not args.no_layer_norm,
        time_embed_dim=args.time_embed_dim,
        target_scale=args.target_scale,
    )
    model = GuidanceNet(cfg, image_encoders).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 1
    if args.resume_from:
        loaded_model, _, extra = load_guidance_checkpoint(
            args.resume_from, image_encoders, device=args.device, load_optimizer=True
        )
        model.load_state_dict(loaded_model.state_dict())
        if extra.get("optimizer_state"):
            optim.load_state_dict(extra["optimizer_state"])
        start_step = (extra.get("step") or 0) + 1
        print(f"resumed from {args.resume_from} at step {start_step - 1}")

    log = []
    final_step = start_step + args.n_steps - 1

    for step in range(start_step, final_step + 1):
        if args.overfit_first_batch:
            target_batch = first_target
        else:
            batch, iterator = next_agf_batch(iterator, loader, policy, args.device)
            target_batch = compute_targets(policy, critic, model, batch, args)

        set_trainable(model, True)

        image = target_batch["image"].to(args.device)
        image2 = target_batch["image2"].to(args.device)
        x_t = target_batch["x_t"].to(args.device)
        t = target_batch["t"].to(args.device)
        target = target_batch["adjoint_target"].to(args.device)

        pred = model(image, image2, x_t, t)
        loss = region_mse(pred, target, spec)

        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optim.step()

        last_metrics = None
        if step == start_step or step % args.log_every == 0 or step == final_step:
            adj = target_batch["adjoint_norm"].float()
            b = args.batch_size
            extra = {
                "seed_norm": float(target_batch["seed_norm"].float().mean().cpu()),
                "adjoint_norm": float(adj.mean().cpu()),
                "adjoint_norm_s0": float(adj[:b].mean().cpu()),
                "q_value": float(target_batch["q_value"].float().mean().cpu()),
                "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            }
            last_metrics = metric_row(step, loss, pred, target, spec, extra)
            log.append(last_metrics)
            save_log_and_plot(log, args.output_dir)

        if args.save_every and step % args.save_every == 0:
            if last_metrics is None:
                adj = target_batch["adjoint_norm"].float()
                b = args.batch_size
                extra = {
                    "seed_norm": float(target_batch["seed_norm"].float().mean().cpu()),
                    "adjoint_norm": float(adj.mean().cpu()),
                    "adjoint_norm_s0": float(adj[:b].mean().cpu()),
                    "q_value": float(target_batch["q_value"].float().mean().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                }
                last_metrics = metric_row(step, loss, pred, target, spec, extra)
            save_guidance_checkpoint(
                os.path.join(args.output_dir, f"ckpt_step_{step}.pt"),
                model,
                cfg,
                optimizer=optim,
                step=step,
                metrics=last_metrics,
            )

    save_training_outputs(args, model, cfg, optim, log, "agf_discrete_guidance.pt")
    return model


def train_pi05_robocasa(args):
    from critics.robocasa import load_critic
    from guidance.networks_robocasa import (
        GuidanceConfig,
        GuidanceNet,
        image_encoders_from_critic,
        load_guidance_checkpoint,
        save_guidance_checkpoint,
    )
    from guidance.targets_pi05_robocasa import ActionSpec
    from training.pi05_robocasa import (
        agf_targets as compute_targets,
        make_agf_dataloader,
        metric_row,
        next_agf_batch,
        region_mse,
        save_log_and_plot,
        save_training_outputs,
        set_trainable,
    )
    import os
    import random
    import torch

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from lerobot.policies.pi05 import PI05Policy

    loader = make_agf_dataloader(args)
    iterator = iter(loader)

    policy = PI05Policy.from_pretrained(args.pretrained_path).to(args.device).eval()
    critic = load_critic(args.critic_ckpt, args.device).eval()

    for module in (policy, critic):
        for p in module.parameters():
            p.requires_grad_(False)

    image_encoders = image_encoders_from_critic(critic, ensemble_size=args.ensemble_size)

    first_batch, iterator = next_agf_batch(iterator, loader, args.device)
    first_target = compute_targets(policy, critic, None, first_batch, args, first_batch=True)
    spec = ActionSpec(args.critic_horizon, args.action_dim)

    cfg = GuidanceConfig(
        guidance_type="agf_discrete",
        action_dim=args.action_dim,
        critic_horizon=args.critic_horizon,
        chunk_size=first_target["x_t"].shape[1],
        max_action_dim=first_target["x_t"].shape[2],
        state_dim=first_target["state"].shape[-1],
        image_feature_dim=args.image_feature_dim,
        ensemble_size=args.ensemble_size,
        hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",")),
        layer_norm=not args.no_layer_norm,
        time_embed_dim=args.time_embed_dim,
        target_scale=args.target_scale,
    )
    model = GuidanceNet(cfg, image_encoders).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 1
    if args.resume_from:
        loaded_model, _, extra = load_guidance_checkpoint(
            args.resume_from, image_encoders, device=args.device, load_optimizer=True
        )
        model.load_state_dict(loaded_model.state_dict())
        if extra.get("optimizer_state"):
            optim.load_state_dict(extra["optimizer_state"])
        start_step = (extra.get("step") or 0) + 1
        print(f"resumed from {args.resume_from} at step {start_step - 1}")

    log = []
    final_step = start_step + args.n_steps - 1

    for step in range(start_step, final_step + 1):
        if args.overfit_first_batch:
            target_batch = first_target
        else:
            batch, iterator = next_agf_batch(iterator, loader, args.device)
            target_batch = compute_targets(policy, critic, model, batch, args)

        set_trainable(model, True)

        image = target_batch["image"].to(args.device)
        image2 = target_batch["image2"].to(args.device)
        image3 = target_batch["image3"].to(args.device)
        state = target_batch["state"].to(args.device)
        x_t = target_batch["x_t"].to(args.device)
        t = target_batch["t"].to(args.device)
        target = target_batch["adjoint_target"].to(args.device)

        pred = model(image, image2, image3, state, x_t, t)
        loss = region_mse(pred, target, spec)

        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optim.step()

        last_metrics = None
        if step == start_step or step % args.log_every == 0 or step == final_step:
            adj = target_batch["adjoint_norm"].float()
            b = args.batch_size
            extra = {
                "seed_norm": float(target_batch["seed_norm"].float().mean().cpu()),
                "adjoint_norm": float(adj.mean().cpu()),
                "adjoint_norm_s0": float(adj[:b].mean().cpu()),
                "q_value": float(target_batch["q_value"].float().mean().cpu()),
                "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            }
            last_metrics = metric_row(step, loss, pred, target, spec, extra)
            log.append(last_metrics)
            save_log_and_plot(log, args.output_dir)

        if args.save_every and step % args.save_every == 0:
            if last_metrics is None:
                adj = target_batch["adjoint_norm"].float()
                b = args.batch_size
                extra = {
                    "seed_norm": float(target_batch["seed_norm"].float().mean().cpu()),
                    "adjoint_norm": float(adj.mean().cpu()),
                    "adjoint_norm_s0": float(adj[:b].mean().cpu()),
                    "q_value": float(target_batch["q_value"].float().mean().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                }
                last_metrics = metric_row(step, loss, pred, target, spec, extra)
            save_guidance_checkpoint(
                os.path.join(args.output_dir, f"ckpt_step_{step}.pt"),
                model,
                cfg,
                optimizer=optim,
                step=step,
                metrics=last_metrics,
            )

    save_training_outputs(args, model, cfg, optim, log, "agf_discrete_guidance.pt")
    return model


def train_molmoact2(args):
    from critics.libero import load_critic
    from guidance.networks_libero import (
        GuidanceConfig,
        GuidanceNet,
        image_encoders_from_critic,
        load_guidance_checkpoint,
        save_guidance_checkpoint,
    )
    from guidance.targets_molmoact2 import ActionSpec, compute_molmoact2_agf_adjoint_targets
    from training.molmoact2 import (
        make_agf_dataloader,
        metric_row,
        next_agf_batch,
        region_mse,
        save_log_and_plot,
        save_training_outputs,
        set_trainable,
    )
    import os
    import random
    import torch

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from lerobot.policies.molmoact2 import MolmoAct2Policy

    loader = make_agf_dataloader(args)
    iterator = iter(loader)

    policy = MolmoAct2Policy.from_pretrained(args.pretrained_path).to(args.device).eval()
    policy.config.inference_action_mode = "continuous"
    critic = load_critic(args.critic_ckpt, args.device).eval()

    for module in (policy, critic):
        for p in module.parameters():
            p.requires_grad_(False)

    image_encoders = image_encoders_from_critic(critic, ensemble_size=args.ensemble_size)

    first_batch, iterator = next_agf_batch(iterator, loader, args.device)
    first_target = compute_molmoact2_agf_adjoint_targets(
        policy, critic, None, first_batch, args, use_guidance_rollout=False
    )
    spec = ActionSpec(args.critic_horizon, args.action_dim)

    cfg = GuidanceConfig(
        guidance_type="agf_discrete",
        action_dim=args.action_dim,
        critic_horizon=args.critic_horizon,
        chunk_size=first_target["x_t"].shape[1],
        max_action_dim=first_target["x_t"].shape[2],
        image_feature_dim=args.image_feature_dim,
        ensemble_size=args.ensemble_size,
        hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",")),
        layer_norm=not args.no_layer_norm,
        time_embed_dim=args.time_embed_dim,
        target_scale=args.target_scale,
    )
    model = GuidanceNet(cfg, image_encoders).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 1
    if args.resume_from:
        loaded_model, _, extra = load_guidance_checkpoint(
            args.resume_from, image_encoders, device=args.device, load_optimizer=True
        )
        model.load_state_dict(loaded_model.state_dict())
        if extra.get("optimizer_state"):
            optim.load_state_dict(extra["optimizer_state"])
        start_step = (extra.get("step") or 0) + 1
        print(f"resumed from {args.resume_from} at step {start_step - 1}")

    log = []
    final_step = start_step + args.n_steps - 1
    n_accum = args.grad_accum_steps

    for step in range(start_step, final_step + 1):
        set_trainable(model, True)
        optim.zero_grad()

        micro_preds, micro_targets = [], []
        micro_seed_norms, micro_adjoint_norms, micro_adjoint_norm_s0, micro_q_values = [], [], [], []
        accum_loss = 0.0

        for _ in range(n_accum):
            if args.overfit_first_batch:
                target_batch = first_target
            else:
                batch, iterator = next_agf_batch(iterator, loader, args.device)
                target_batch = compute_molmoact2_agf_adjoint_targets(policy, critic, model, batch, args)

            image = target_batch["image"].to(args.device)
            image2 = target_batch["image2"].to(args.device)
            x_t = target_batch["x_t"].to(args.device)
            t = target_batch["t"].to(args.device)
            target = target_batch["adjoint_target"].to(args.device)

            pred = model(image, image2, x_t, t)
            loss = region_mse(pred, target, spec)
            (loss / n_accum).backward()
            accum_loss += float(loss.detach().cpu()) / n_accum

            micro_preds.append(pred.detach())
            micro_targets.append(target.detach())
            adj = target_batch["adjoint_norm"].float()
            micro_seed_norms.append(float(target_batch["seed_norm"].float().mean().cpu()))
            micro_adjoint_norms.append(float(adj.mean().cpu()))

            micro_adjoint_norm_s0.append(float(adj[: args.batch_size].mean().cpu()))
            micro_q_values.append(float(target_batch["q_value"].float().mean().cpu()))

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optim.step()

        last_metrics = None
        if step == start_step or step % args.log_every == 0 or step == final_step:
            extra = {
                "seed_norm": sum(micro_seed_norms) / n_accum,
                "adjoint_norm": sum(micro_adjoint_norms) / n_accum,
                "adjoint_norm_s0": sum(micro_adjoint_norm_s0) / n_accum,
                "q_value": sum(micro_q_values) / n_accum,
                "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            }
            last_metrics = metric_row(
                step, torch.tensor(accum_loss), torch.cat(micro_preds), torch.cat(micro_targets), spec, extra
            )
            log.append(last_metrics)
            save_log_and_plot(log, args.output_dir)

        if args.save_every and step % args.save_every == 0:
            if last_metrics is None:
                extra = {
                    "seed_norm": sum(micro_seed_norms) / n_accum,
                    "adjoint_norm": sum(micro_adjoint_norms) / n_accum,
                    "adjoint_norm_s0": sum(micro_adjoint_norm_s0) / n_accum,
                    "q_value": sum(micro_q_values) / n_accum,
                    "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                }
                last_metrics = metric_row(
                    step, torch.tensor(accum_loss), torch.cat(micro_preds), torch.cat(micro_targets), spec, extra
                )
            save_guidance_checkpoint(
                os.path.join(args.output_dir, f"ckpt_step_{step}.pt"),
                model,
                cfg,
                optimizer=optim,
                step=step,
                metrics=last_metrics,
            )

    save_training_outputs(args, model, cfg, optim, log, "agf_discrete_guidance.pt")
    return model


def main(argv=None):
    import argparse
    import sys
    import options

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description="Train AGF. Select --vla and --env; add --help for training options.",
    )
    parser.add_argument("--vla", choices=("smolvla", "pi05", "molmoact2"), required=True)
    parser.add_argument("--env", choices=("libero", "libero_pro", "robocasa"), required=True)
    if not argv or (
        any(x in argv for x in ("-h", "--help"))
        and not all(any(x == flag or x.startswith(flag + "=") for x in argv) for flag in ("--vla", "--env"))
    ):
        parser.print_help()
        print("\nSmolVLA/MolmoAct2: libero, libero_pro. pi05: libero, libero_pro, robocasa.")
        return
    selection, remaining = parser.parse_known_args(argv)
    if selection.env == "robocasa" and selection.vla != "pi05":
        parser.error("RoboCasa AGF training is supported only for pi05")

    if selection.vla == "smolvla":
        return train_smolvla(options.agf_smolvla(remaining))
    if selection.vla == "molmoact2":
        return train_molmoact2(options.agf_molmoact2(remaining))
    if selection.env == "robocasa":
        return train_pi05_robocasa(options.agf_pi05_robocasa(remaining))
    return train_pi05_libero(options.agf_pi05_libero(remaining))


if __name__ == "__main__":
    main()
