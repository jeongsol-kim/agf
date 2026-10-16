from __future__ import annotations

import torch


def corrcoef(x, y):
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if denom <= 0:
        return float("nan")
    return float((x * y).mean().div(denom).item())


def compute_oracle(dataset, gamma):
    rewards = []
    valid = []
    executed_steps = []

    for i, record in enumerate(dataset.records):
        success_step = record.get("success_step")
        step = int(record.get("step", 0))
        n = int(record.get("executed_steps", dataset.action_horizon))
        reward = float(success_step is not None and step <= int(success_step) < step + n)

        rewards.append(reward)
        valid.append(float(dataset.next_indices[i] is not None))
        executed_steps.append(float(n))

    oracle = [0.0] * len(dataset)
    for i in reversed(range(len(dataset))):
        next_i = dataset.next_indices[i]
        if next_i is None:
            oracle[i] = rewards[i]
        else:
            oracle[i] = rewards[i] + (gamma ** executed_steps[i]) * oracle[next_i]

    return (
        torch.tensor(oracle, dtype=torch.float32),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(valid, dtype=torch.float32),
    )


def evaluate_q(args):
    from critics.libero import load_critic
    from tasks.datasets import LiberoCriticDataTaskDataset
    from torch.utils.data import DataLoader
    import json
    import os

    critic = load_critic(args.ckpt, device=args.device)
    critic.eval()
    cfg = critic.cfg

    dataset = LiberoCriticDataTaskDataset(
        critic_data_root=args.critic_data_root,
        suite=args.suite,
        task_id=args.task_id,
        action_horizon=cfg.action_horizon,
        action_dim=cfg.action_dim,
        image2_key=args.image2_key,
    )
    oracle_q, rewards, valid = compute_oracle(dataset, cfg.gamma)

    if args.max_samples is not None:
        n = min(args.max_samples, len(dataset))
        indices = list(range(n))
        oracle_q = oracle_q[:n]
        rewards = rewards[:n]
        valid = valid[:n]
        eval_dataset = torch.utils.data.Subset(dataset, indices)
    else:
        eval_dataset = dataset

    loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=False
    )

    q_means = []
    q_pess = []
    with torch.no_grad():
        net = critic.q_target if args.use_target else critic.q
        for batch in loader:
            image = batch["image"].to(args.device)
            image2 = batch["image2"].to(args.device)
            action = batch["action"].to(args.device)

            q = net(image, image2, action)
            mean = q.mean(dim=-1)
            std = q.std(dim=-1, unbiased=False)
            q_means.append(mean.cpu())
            q_pess.append((mean - cfg.rho * std).cpu())

    q_mean = torch.cat(q_means)
    q_pess = torch.cat(q_pess)

    terminal = valid < 0.5
    terminal_success = terminal & (rewards > 0.5)
    terminal_failure = terminal & (rewards < 0.5)

    metrics = {
        "ckpt": args.ckpt,
        "suite": args.suite,
        "task_id": args.task_id,
        "n": int(q_mean.numel()),
        "reward_mean": float(rewards.mean().item()),
        "done_mean": float((1.0 - valid).mean().item()),
        "oracle_q_mean": float(oracle_q.mean().item()),
        "q_mean": float(q_mean.mean().item()),
        "q_mse": float(torch.mean((q_mean - oracle_q) ** 2).item()),
        "q_corr": corrcoef(q_mean, oracle_q),
        "q_pess_corr": corrcoef(q_pess, oracle_q),
        "terminal_success_n": int(terminal_success.sum().item()),
        "terminal_failure_n": int(terminal_failure.sum().item()),
        "oracle_terminal_mean": float(oracle_q[terminal].mean().item()),
        "q_terminal_mean": float(q_mean[terminal].mean().item()),
        "q_terminal_success_mean": float(q_mean[terminal_success].mean().item()),
        "q_terminal_failure_mean": float(q_mean[terminal_failure].mean().item()),
        "q_terminal_gap": float(q_mean[terminal_success].mean().sub(q_mean[terminal_failure].mean()).item()),
    }

    print(json.dumps(metrics, indent=2))

    output_path = args.output_path
    if output_path is None:
        ckpt_dir = os.path.dirname(args.ckpt) or "."
        output_path = os.path.join(ckpt_dir, f"eval_q_{args.suite}_task{args.task_id:02d}.json")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved: {output_path}")


def main(argv=None):
    import options

    args = options.eval_critic(argv)
    return evaluate_q(args)


if __name__ == "__main__":
    main()
