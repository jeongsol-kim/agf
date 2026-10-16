from __future__ import annotations


def train_libero(args):
    from critics.libero import Config, QPILOTSLiberoCritic, _LOG_KEYS, plot_log, save_log, set_lr
    from dataclasses import asdict
    from tasks.datasets import make_critic_data_task_dataloader, make_libero_q_dataloader
    import torch.nn as nn
    import numpy as np
    import os
    import torch

    cfg = Config(n_steps=args.n_steps, batch_size=args.batch_size, num_workers=args.num_workers)
    print(cfg)
    output_dir, device = args.output_dir, args.device
    os.makedirs(output_dir, exist_ok=True)

    if args.data_type == "offpolicy":
        _, loader = make_libero_q_dataloader(
            args.data_path,
            action_horizon=cfg.action_horizon,
            action_dim=cfg.action_dim,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            task_id=args.task_id,
        )
    else:
        if args.task_id is None:
            raise ValueError("--task_id is required when --type onpolicy")
        _, loader = make_critic_data_task_dataloader(
            critic_data_root=args.critic_data_root,
            suite=args.suite,
            task_id=args.task_id,
            action_horizon=cfg.action_horizon,
            action_dim=cfg.action_dim,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            image2_key=args.image2_key,
        )
    iterator = iter(loader)

    critic = QPILOTSLiberoCritic(cfg).to(device)
    optimizer = torch.optim.Adam(critic.q.parameters(), lr=cfg.lr)

    log = {k: [] for k in _LOG_KEYS}
    log_avg = {k: [] for k in _LOG_KEYS}

    for step in range(1, cfg.n_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        lr = set_lr(optimizer, step, cfg)
        loss, info = critic.loss(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(critic.q.parameters(), cfg.grad_clip)
        optimizer.step()
        critic.update_target()

        info["grad_norm"] = float(grad_norm)
        info["lr"] = float(lr)
        for k in _LOG_KEYS:
            log[k].append(info[k])

        if step % cfg.log_every == 0:
            avg = {k: float(np.nanmean(log[k])) for k in _LOG_KEYS}
            for k in _LOG_KEYS:
                log_avg[k].append(avg[k])
                log[k].clear()
            print(
                f"step {step:7d} | loss {avg['q_loss']:.4f} | td {avg['td_abs']:.3f} | "
                f"r {avg['reward_mean']:.3f}/{avg['reward_max']:.1f} | "
                f"done {avg['done_mean']:.3f} | q {avg['q_mean']:.2f} | "
                f"tgt {avg['target_mean']:.2f} | q_term {avg['q_terminal_mean']:.2f} | "
                f"q_non {avg['q_nonterminal_mean']:.2f} | grad {avg['grad_norm']:.2f} | "
                f"lr {avg['lr']:.1e}"
            )
            save_log(log_avg, output_dir)
            plot_log(log_avg, output_dir)

        if step % cfg.save_every == 0 or step == cfg.n_steps:
            ckpt_path = os.path.join(output_dir, f"qpilots_libero_q_step{step:06d}.pt")
            torch.save(
                {
                    "step": step,
                    "cfg": asdict(cfg),
                    "q": critic.q.state_dict(),
                    "q_target": critic.q_target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"saved: {ckpt_path}")

    save_log(log_avg, output_dir)
    plot_log(log_avg, output_dir)

    return critic


def train_robocasa(args):
    from critics.robocasa import Config, QPILOTSRoboCasaCritic, _LOG_KEYS, plot_log, save_log, set_lr
    from dataclasses import asdict
    from tasks.datasets import make_robocasa_critic_dataloader
    import torch.nn as nn
    import numpy as np
    import os
    import torch

    cfg = Config(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_dims=(args.hidden_dim,) * 3,
        image_size=args.image_size,
    )
    print(cfg)
    output_dir, device = args.output_dir, args.device
    os.makedirs(output_dir, exist_ok=True)

    _, loader = make_robocasa_critic_dataloader(
        critic_data_root=args.critic_data_root,
        group_name=args.group_name,
        task_name=args.task_name,
        action_horizon=cfg.action_horizon,
        action_dim=cfg.action_dim,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    iterator = iter(loader)

    critic = QPILOTSRoboCasaCritic(cfg).to(device)
    optimizer = torch.optim.Adam(critic.q.parameters(), lr=cfg.lr)

    log = {k: [] for k in _LOG_KEYS}
    log_avg = {k: [] for k in _LOG_KEYS}

    for step in range(1, cfg.n_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        lr = set_lr(optimizer, step, cfg)
        loss, info = critic.loss(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(critic.q.parameters(), cfg.grad_clip)
        optimizer.step()
        critic.update_target()

        info["grad_norm"] = float(grad_norm)
        info["lr"] = float(lr)
        for k in _LOG_KEYS:
            log[k].append(info[k])

        if step % cfg.log_every == 0:
            avg = {k: float(np.nanmean(log[k])) for k in _LOG_KEYS}
            for k in _LOG_KEYS:
                log_avg[k].append(avg[k])
                log[k].clear()
            print(
                f"step {step:7d} | loss {avg['q_loss']:.4f} | td {avg['td_abs']:.3f} | "
                f"r {avg['reward_mean']:.3f}/{avg['reward_max']:.1f} | "
                f"done {avg['done_mean']:.3f} | q {avg['q_mean']:.2f} | "
                f"tgt {avg['target_mean']:.2f} | q_term {avg['q_terminal_mean']:.2f} | "
                f"q_non {avg['q_nonterminal_mean']:.2f} | grad {avg['grad_norm']:.2f} | "
                f"lr {avg['lr']:.1e}"
            )
            save_log(log_avg, output_dir)
            plot_log(log_avg, output_dir)

        if step % cfg.save_every == 0 or step == cfg.n_steps:
            ckpt_path = os.path.join(output_dir, f"qpilots_robocasa_q_step{step:06d}.pt")
            torch.save(
                {
                    "step": step,
                    "cfg": asdict(cfg),
                    "q": critic.q.state_dict(),
                    "q_target": critic.q_target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"saved: {ckpt_path}")

    save_log(log_avg, output_dir)
    plot_log(log_avg, output_dir)

    return critic


def main(argv=None):
    import argparse
    import sys
    import options

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description="Train Q from saved data. Select --env; add --help for training options.",
    )
    parser.add_argument("--env", choices=("libero", "libero_pro", "robocasa"), required=True)
    if not argv or (
        any(x in argv for x in ("-h", "--help")) and not any(x == "--env" or x.startswith("--env=") for x in argv)
    ):
        parser.print_help()
        return
    selection, remaining = parser.parse_known_args(argv)
    if selection.env == "robocasa":
        return train_robocasa(options.critic_robocasa(remaining))
    return train_libero(options.critic_libero(remaining))


if __name__ == "__main__":
    main()
