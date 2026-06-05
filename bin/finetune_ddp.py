#!/usr/bin/env python3
"""Multi-GPU DDP fine-tuning for MethylBERT read classification.

Launched with ``torchrun`` (one process per GPU). Reuses the official
``MethylBertEmbeddedDMR`` model and ``MethylVocab`` tokenizer via ``mb_lib`` and
reads pre-tokenized parquet shards (no re-tokenization, no whole-file loads).

Features (per the refactor spec):
  * reads LOCAL_RANK / RANK / WORLD_SIZE, ``torch.cuda.set_device(local_rank)``
  * ``DistributedSampler`` for train + validation
  * loss reduced across ranks for logging; validation metrics gathered across
    all ranks (not computed from a single rank)
  * mixed precision (``--precision fp16|bf16|fp32``)
  * gradient accumulation (``--gradient-accumulation-steps``)
  * fixed seed, resume from checkpoint
  * low-frequency validation (per epoch, or every ``--val-every-steps`` opt steps)
  * AUC / PR-AUC / balanced accuracy + a per-checkpoint metric table
  * checkpoints / best model / metrics / plots written on rank 0 only
"""

import argparse
import csv
import os
import sys
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import mb_lib  # noqa: E402


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
def ddp_env():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_main():
    return (not is_dist()) or dist.get_rank() == 0


def reduce_mean(value: float, device) -> float:
    if not is_dist():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def gather_arrays(arr: np.ndarray):
    """All-gather 1-D numpy arrays of differing lengths to every rank."""
    if not is_dist():
        return arr
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, arr)
    return np.concatenate([g for g in gathered if g is not None and len(g)])


def log_main(*a):
    if is_main():
        print(*a, flush=True)


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-shards", required=True)
    p.add_argument("--val-shards", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--dmr", required=True, help="filtered_dmr.bed (defines n_dmrs)")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=64, help="per-GPU batch size")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.06,
                   help="Linear LR warmup as a fraction of total optimizer steps")
    p.add_argument("--decay-start-ratio", type=float, default=0.90,
                   help="Optimizer step (as fraction of total) where linear LR decay begins; decay runs to the end")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-shards", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=mb_lib.DEFAULT_SEQ_LEN)
    p.add_argument("--loss", default="bce", choices=["bce", "focal_bce"])
    p.add_argument("--val-every-steps", type=int, default=0,
                   help="optimizer steps between validations; 0 = once per epoch")
    p.add_argument("--log-every-steps", type=int, default=10)
    p.add_argument("--best-metric", default="auc",
                   choices=["auc", "pr_auc", "balanced_accuracy", "accuracy", "loss"])
    p.add_argument("--resume", default="", help="path to training_state.pt")
    return p.parse_args()


def n_dmrs_from_bed(path: str) -> int:
    mx = -1
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            mx = max(mx, int(line.split("\t")[3]))
    return mx + 1


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, loader, device, precision):
    amp_enabled, amp_dtype, _ = mb_lib.resolve_amp(precision)
    model.eval()
    losses, probs, preds, labels = [], [], [], []
    for batch in loader:
        ids = batch["dna_seq"].to(device, non_blocking=True)
        tt = batch["methyl_seq"].to(device, non_blocking=True)
        dmr = batch["dmr_label"].to(device, non_blocking=True)
        ctype = batch["ctype_label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=amp_dtype, enabled=amp_enabled):
            out = model(step=0, input_ids=ids, token_type_ids=tt,
                        labels=dmr, ctype_label=ctype)
        losses.append(float(out["loss"].mean().item()))
        cl = out["classification_logits"].float().cpu().numpy()
        probs.append(cl[:, 1])
        preds.append(cl.argmax(axis=1))
        labels.append(batch["ctype_label"].numpy())
    model.train()

    local_loss = float(np.mean(losses)) if losses else 0.0
    loss = reduce_mean(local_loss, device)
    y_prob = gather_arrays(np.concatenate(probs) if probs else np.array([]))
    y_pred = gather_arrays(np.concatenate(preds) if preds else np.array([], dtype=int))
    y_true = gather_arrays(np.concatenate(labels) if labels else np.array([], dtype=int))

    metrics = {"loss": loss}
    metrics.update(mb_lib.classification_metrics(y_true, y_prob, y_pred))
    return metrics


def better(metric_name, new, best):
    if best is None:
        return True
    if metric_name == "loss":
        return new < best
    return new > best


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_plots(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        log_main(f"Skipping plots ({exc})")
        return
    if not rows:
        return
    steps = [r["global_step"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, [r["train_loss"] for r in rows], label="train_loss")
    ax[0].plot(steps, [r["val_loss"] for r in rows], label="val_loss")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(steps, [r["val_auc"] for r in rows], label="val_auc")
    ax[1].plot(steps, [r["val_pr_auc"] for r in rows], label="val_pr_auc")
    ax[1].plot(steps, [r["val_balanced_accuracy"] for r in rows], label="val_bal_acc")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("metric"); ax[1].legend()
    ax[1].set_title("Validation metrics")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=120)
    plt.close(fig)


METRIC_FIELDS = ["epoch", "global_step", "lr", "train_loss", "val_loss",
                 "val_accuracy", "val_balanced_accuracy", "val_auc", "val_pr_auc",
                 "val_n", "checkpoint", "is_best"]


def write_metric_table(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in METRIC_FIELDS})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    rank, local_rank, world_size = ddp_env()

    if world_size > 1:
        # Generous timeout so slow rank-0 checkpoint I/O (Lustre) during the
        # save/barrier window never trips the default 10-minute NCCL watchdog.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # Reproducibility (rank offset keeps per-rank RNG distinct but deterministic).
    from methylbert.utils import set_seed
    set_seed(args.seed + rank)

    os.makedirs(args.output, exist_ok=True)
    n_dmrs = n_dmrs_from_bed(args.dmr)
    log_main(f"World size: {world_size} | device: {device} | n_dmrs: {n_dmrs} | "
             f"precision: {args.precision}")

    # Datasets / loaders.
    train_ds = mb_lib.ShardDataset(args.train_shards, seq_len=args.seq_len,
                                   cache_shards=args.cache_shards)
    val_ds = mb_lib.ShardDataset(args.val_shards, seq_len=args.seq_len,
                                 cache_shards=args.cache_shards)
    log_main(f"Train reads: {len(train_ds)} | Val reads: {len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False,
                                       seed=args.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) \
        if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=args.num_workers,
        pin_memory=True, drop_last=False, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0)

    # Model (official class) + DDP.
    model = mb_lib.build_pretrained_model(args.pretrained, n_dmrs,
                                          seq_len=args.seq_len, loss=args.loss)
    model.to(device)
    if world_size > 1:
        # The BERT pooler is built by BertModel but its output is unused by the
        # read-classification head, so those params get no gradient -> DDP needs
        # find_unused_parameters=True.
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), eps=1e-6,
                                  weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    total_opt_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_opt_steps * args.warmup_ratio))
    decrease_steps = max(warmup_steps + 1,
                         min(total_opt_steps - 1,
                             int(total_opt_steps * args.decay_start_ratio)))
    from methylbert.trainer import learning_rate_scheduler
    scheduler = learning_rate_scheduler(optimizer, num_warmup_steps=warmup_steps,
                                        num_training_steps=total_opt_steps,
                                        decrease_steps=decrease_steps)
    log_main(f"Optimizer steps/epoch: {steps_per_epoch} | total: {total_opt_steps} | "
             f"LR warmup: {warmup_steps} steps ({args.warmup_ratio:.0%}) | "
             f"LR decay from step: {decrease_steps} ({args.decay_start_ratio:.0%})")

    amp_enabled, amp_dtype, use_scaler = mb_lib.resolve_amp(args.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Resume.
    start_epoch = 0
    global_step = 0
    best_value = None
    metric_rows = []
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if hasattr(model, "module") else model).load_state_dict(
            ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_value = ckpt.get("best_value", None)
        metric_rows = ckpt.get("metric_rows", [])
        log_main(f"Resumed from {args.resume} at epoch {start_epoch}, step {global_step}")

    best_dir = os.path.join(args.output, "bert.model")
    metric_path = os.path.join(args.output, "metrics_per_checkpoint.csv")
    train_log_path = os.path.join(args.output, "train_log.csv")
    if is_main() and not os.path.exists(train_log_path):
        with open(train_log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "global_step", "train_loss", "lr"])

    def run_validation_and_checkpoint(epoch):
        nonlocal best_value
        metrics = validate(model, val_loader, device, args.precision)
        ckpt_name = f"bert.model_step{global_step}"
        is_best = better(args.best_metric, metrics[args.best_metric], best_value)
        if is_best:
            best_value = metrics[args.best_metric]
        if is_main():
            step_dir = os.path.join(args.output, ckpt_name)
            mb_lib.save_finetuned_model(model, step_dir)
            if is_best:
                mb_lib.save_finetuned_model(model, best_dir)
            row = {
                "epoch": epoch, "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": running_train_loss[0],
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_auc": metrics["auc"], "val_pr_auc": metrics["pr_auc"],
                "val_n": metrics["n"], "checkpoint": ckpt_name,
                "is_best": int(is_best),
            }
            metric_rows.append(row)
            write_metric_table(metric_rows, metric_path)
            save_plots(metric_rows, args.output)
            log_main(f"[val] epoch {epoch} step {global_step} | "
                     f"loss {metrics['loss']:.4f} acc {metrics['accuracy']:.4f} "
                     f"bal_acc {metrics['balanced_accuracy']:.4f} "
                     f"auc {metrics['auc']:.4f} pr_auc {metrics['pr_auc']:.4f} "
                     f"(best {args.best_metric}={best_value})")
        if is_dist():
            dist.barrier()

    running_train_loss = [0.0]
    model.train()
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        micro = 0
        for batch in train_loader:
            ids = batch["dna_seq"].to(device, non_blocking=True)
            tt = batch["methyl_seq"].to(device, non_blocking=True)
            dmr = batch["dmr_label"].to(device, non_blocking=True)
            ctype = batch["ctype_label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=amp_dtype, enabled=amp_enabled):
                out = model(step=global_step, input_ids=ids, token_type_ids=tt,
                            labels=dmr, ctype_label=ctype)
                loss = out["loss"].mean() / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            micro += 1

            if micro % args.gradient_accumulation_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                running_train_loss[0] = reduce_mean(accum_loss, device)

                if is_main() and global_step % args.log_every_steps == 0:
                    with open(train_log_path, "a", newline="") as fh:
                        csv.writer(fh).writerow([
                            epoch, global_step, f"{running_train_loss[0]:.6f}",
                            f"{optimizer.param_groups[0]['lr']:.3e}"])
                    log_main(f"[train] epoch {epoch} step {global_step} "
                             f"loss {running_train_loss[0]:.4f} "
                             f"lr {optimizer.param_groups[0]['lr']:.3e}")

                if args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                    run_validation_and_checkpoint(epoch)
                accum_loss = 0.0

        # Low-frequency validation: at minimum once per epoch.
        if args.val_every_steps == 0:
            run_validation_and_checkpoint(epoch)

        # Persist resumable training state (rank 0).
        if is_main():
            torch.save({
                "model": (model.module if hasattr(model, "module") else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_scaler else None,
                "epoch": epoch + 1, "global_step": global_step,
                "best_value": best_value, "metric_rows": metric_rows,
            }, os.path.join(args.output, "training_state.pt"))
        if is_dist():
            dist.barrier()

    if is_main():
        log_main(f"Fine-tuning done. Best {args.best_metric}={best_value}. "
                 f"Best model: {best_dir}")
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
