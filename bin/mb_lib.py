#!/usr/bin/env python3
"""Shared MethylBERT helpers for the Nextflow fine-tune / infer pipelines.

This module is intentionally dependency-light (numpy, pyarrow, torch, sklearn)
and reuses the *official* MethylBERT model/tokenizer classes instead of copying
upstream code:

* ``MethylVocab``              -> k-mer tokenizer / encoding
* ``MethylBertEmbeddedDMR``    -> read-classification model (BERT + DMR encoder)
* ``learning_rate_scheduler``  -> warmup / plateau / decay LR schedule

It provides:

* :func:`encode_read`        - turn a preprocessed (dna_seq, methyl_seq) pair
                               into fixed-length token-id / token-type arrays,
                               replicating ``MethylBertFinetuneDataset.__getitem__``
                               *exactly* so the model sees identical inputs.
* :class:`ShardWriter`       - stream tokenized rows to parquet shards.
* :class:`ShardDataset`      - map-style torch Dataset over parquet shards with a
                               bounded LRU shard cache (lazy; never re-tokenizes,
                               never loads every shard at once).
* model build / save / load wrappers.
* :func:`run_read_level_inference` and :func:`classification_metrics`.
"""

import glob
import math
import os
from collections import OrderedDict

import numpy as np

DEFAULT_SEQ_LEN = 150

# Column names used in tokenized parquet shards.
COL_NAME = "name"
COL_INPUT_IDS = "input_ids"
COL_TOKEN_TYPE = "token_type_ids"
COL_DMR_LABEL = "dmr_label"
COL_CTYPE_LABEL = "ctype_label"


# --------------------------------------------------------------------------- #
# Tokenizer / encoding
# --------------------------------------------------------------------------- #
def get_vocab(k: int = 3):
    """Return the official MethylBERT k-mer vocabulary."""
    from methylbert.data.vocab import MethylVocab

    return MethylVocab(k=k)


def encode_read(dna_seq, methyl_seq, vocab, seq_len: int = DEFAULT_SEQ_LEN):
    """Encode one read into (input_ids, token_type_ids), each of length ``seq_len + 1``.

    Faithfully replicates ``MethylBertFinetuneDataset._line2tokens_finetune`` +
    ``__getitem__`` from MethylBERT 2.0.2 (SOS prepend, EOS marker, padding,
    methyl special value = 2). ``dna_seq`` is a space separated k-mer string and
    ``methyl_seq`` a digit string, both produced by the preprocessing step.
    """
    tokens = dna_seq.split(" ") if isinstance(dna_seq, str) else list(dna_seq)
    ids = vocab.to_seq(tokens)  # stoi.get(kmer, unk) per token
    methyl = [int(m) for m in methyl_seq]

    # k-mer and methyl streams must align; guard against malformed rows.
    n = min(len(ids), len(methyl))
    ids, methyl = ids[:n], methyl[:n]

    if len(ids) > seq_len:
        ids = ids[:seq_len]
        methyl = methyl[:seq_len]
    else:
        pad = seq_len - len(ids)
        ids = ids + [vocab.pad_index] * pad
        methyl = methyl + [2] * pad

    # End of the read = last non-pad position + 1.
    nonpad = np.nonzero(np.asarray(ids) != vocab.pad_index)[0]
    end = int(nonpad[-1]) + 1 if nonpad.size else seq_len
    if end < seq_len:
        ids[end] = vocab.eos_index
        methyl[end] = 2
    else:
        ids[-1] = vocab.eos_index
        methyl[-1] = 2

    ids = [vocab.sos_index] + ids
    methyl = [2] + methyl
    return ids, methyl


# --------------------------------------------------------------------------- #
# Parquet shard writer (streaming)
# --------------------------------------------------------------------------- #
class ShardWriter:
    """Accumulate tokenized rows and flush them to parquet shards on disk.

    Raw sequences are intentionally *not* stored; only model-ready token arrays
    and labels are kept so shards stay small.
    """

    def __init__(self, out_dir: str, prefix: str, shard_size: int = 200_000,
                 seq_len: int = DEFAULT_SEQ_LEN):
        import pyarrow as pa

        self._pa = pa
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_size = int(shard_size)
        self.seq_len = seq_len
        os.makedirs(out_dir, exist_ok=True)

        self._schema = pa.schema([
            (COL_NAME, pa.string()),
            (COL_INPUT_IDS, pa.list_(pa.int16(), seq_len + 1)),
            (COL_TOKEN_TYPE, pa.list_(pa.int16(), seq_len + 1)),
            (COL_DMR_LABEL, pa.int32()),
            (COL_CTYPE_LABEL, pa.int8()),
        ])
        self._reset_buffer()
        self._shard_idx = 0
        self.total_rows = 0

    def _reset_buffer(self):
        self._buf = {
            COL_NAME: [],
            COL_INPUT_IDS: [],
            COL_TOKEN_TYPE: [],
            COL_DMR_LABEL: [],
            COL_CTYPE_LABEL: [],
        }
        self._buf_len = 0

    def add(self, name, input_ids, token_type_ids, dmr_label, ctype_label):
        self._buf[COL_NAME].append("" if name is None else str(name))
        self._buf[COL_INPUT_IDS].append(input_ids)
        self._buf[COL_TOKEN_TYPE].append(token_type_ids)
        self._buf[COL_DMR_LABEL].append(int(dmr_label))
        self._buf[COL_CTYPE_LABEL].append(int(ctype_label))
        self._buf_len += 1
        if self._buf_len >= self.shard_size:
            self.flush()

    def flush(self):
        if self._buf_len == 0:
            return
        pa = self._pa
        flat_ids = np.asarray(self._buf[COL_INPUT_IDS], dtype=np.int16).reshape(-1)
        flat_tt = np.asarray(self._buf[COL_TOKEN_TYPE], dtype=np.int16).reshape(-1)
        n = self._buf_len
        table = pa.table({
            COL_NAME: pa.array(self._buf[COL_NAME], type=pa.string()),
            COL_INPUT_IDS: pa.FixedSizeListArray.from_arrays(
                pa.array(flat_ids, type=pa.int16()), self.seq_len + 1),
            COL_TOKEN_TYPE: pa.FixedSizeListArray.from_arrays(
                pa.array(flat_tt, type=pa.int16()), self.seq_len + 1),
            COL_DMR_LABEL: pa.array(self._buf[COL_DMR_LABEL], type=pa.int32()),
            COL_CTYPE_LABEL: pa.array(self._buf[COL_CTYPE_LABEL], type=pa.int8()),
        }, schema=self._schema)

        import pyarrow.parquet as pq

        path = os.path.join(self.out_dir, f"{self.prefix}_{self._shard_idx:05d}.parquet")
        pq.write_table(table, path, compression="zstd")
        self._shard_idx += 1
        self.total_rows += n
        self._reset_buffer()

    def close(self):
        self.flush()
        return self.total_rows


def tokenize_parquet_to_shards(in_parquet: str, out_dir: str, prefix: str,
                               vocab=None, seq_len: int = DEFAULT_SEQ_LEN,
                               shard_size: int = 200_000, batch_rows: int = 50_000):
    """Stream a preprocessed parquet file into tokenized parquet shards.

    The input is read in row batches (never fully materialised) and each read is
    encoded once. Returns the number of shards and total rows written.
    """
    import pyarrow.parquet as pq

    if vocab is None:
        vocab = get_vocab(3)

    writer = ShardWriter(out_dir, prefix, shard_size=shard_size, seq_len=seq_len)
    pf = pq.ParquetFile(in_parquet)
    cols = ["name", "dna_seq", "methyl_seq", "dmr_label", "ctype_label"]
    available = set(pf.schema_arrow.names)
    use_cols = [c for c in cols if c in available]

    for batch in pf.iter_batches(batch_size=batch_rows, columns=use_cols):
        d = batch.to_pydict()
        names = d.get("name", [None] * batch.num_rows)
        dna = d["dna_seq"]
        methyl = d["methyl_seq"]
        dmr = d["dmr_label"]
        ctype = d.get("ctype_label", [0] * batch.num_rows)
        for i in range(batch.num_rows):
            ids, tt = encode_read(dna[i], methyl[i], vocab, seq_len=seq_len)
            writer.add(names[i], ids, tt, dmr[i], ctype[i])

    total = writer.close()
    return writer._shard_idx, total


# --------------------------------------------------------------------------- #
# Streaming shard dataset
# --------------------------------------------------------------------------- #
def list_shards(path: str):
    """Resolve a shard directory or glob into a sorted list of parquet files."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    else:
        files = sorted(glob.glob(path))
    return files


def _make_shard_dataset_class():
    import torch
    from torch.utils.data import Dataset

    class ShardDataset(Dataset):
        """Map-style dataset over tokenized parquet shards.

        Shards are decoded lazily and cached in a small LRU (default 8 shards) so
        we never hold every shard in memory simultaneously, yet random access for
        ``DistributedSampler`` stays cheap. Compatible with the MethylBERT model
        forward (keys ``dna_seq``, ``methyl_seq``, ``dmr_label``, ``ctype_label``).
        """

        def __init__(self, shards, seq_len=DEFAULT_SEQ_LEN, cache_shards=8):
            import pyarrow.parquet as pq

            if isinstance(shards, str):
                shards = list_shards(shards)
            if not shards:
                raise FileNotFoundError("No parquet shards found.")
            self.files = list(shards)
            self.seq_len = seq_len
            self.cache_shards = max(1, int(cache_shards))

            self.row_counts = [pq.ParquetFile(f).metadata.num_rows for f in self.files]
            self.cum = np.concatenate([[0], np.cumsum(self.row_counts)]).astype(np.int64)
            self._n = int(self.cum[-1])
            self._cache = OrderedDict()  # shard_idx -> dict of numpy arrays

        def __len__(self):
            return self._n

        def num_dmrs(self):
            """Max dmr_label + 1 across all shards (matches official semantics)."""
            import pyarrow.parquet as pq

            mx = -1
            for f in self.files:
                col = pq.read_table(f, columns=[COL_DMR_LABEL])[COL_DMR_LABEL]
                if len(col):
                    mx = max(mx, int(np.asarray(col).max()))
            return mx + 1

        def _load_shard(self, s):
            if s in self._cache:
                self._cache.move_to_end(s)
                return self._cache[s]
            import pyarrow.parquet as pq

            t = pq.read_table(self.files[s])
            data = {
                COL_INPUT_IDS: np.asarray(
                    t[COL_INPUT_IDS].combine_chunks().values, dtype=np.int64
                ).reshape(-1, self.seq_len + 1),
                COL_TOKEN_TYPE: np.asarray(
                    t[COL_TOKEN_TYPE].combine_chunks().values, dtype=np.int64
                ).reshape(-1, self.seq_len + 1),
                COL_DMR_LABEL: np.asarray(t[COL_DMR_LABEL], dtype=np.int64),
                COL_CTYPE_LABEL: np.asarray(t[COL_CTYPE_LABEL], dtype=np.int64),
                COL_NAME: [str(x) for x in t[COL_NAME].to_pylist()],
            }
            self._cache[s] = data
            self._cache.move_to_end(s)
            while len(self._cache) > self.cache_shards:
                self._cache.popitem(last=False)
            return data

        def __getitem__(self, idx):
            s = int(np.searchsorted(self.cum, idx, side="right") - 1)
            local = idx - int(self.cum[s])
            data = self._load_shard(s)
            return {
                "dna_seq": torch.from_numpy(data[COL_INPUT_IDS][local].copy()),
                "methyl_seq": torch.from_numpy(data[COL_TOKEN_TYPE][local].copy()),
                "dmr_label": torch.tensor(int(data[COL_DMR_LABEL][local]), dtype=torch.long),
                "ctype_label": torch.tensor(int(data[COL_CTYPE_LABEL][local]), dtype=torch.long),
                "name": data[COL_NAME][local],
            }

    return ShardDataset


def ShardDataset(*args, **kwargs):  # noqa: N802 (factory keeps torch import lazy)
    cls = _make_shard_dataset_class()
    return cls(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Model build / save / load (reuse official classes)
# --------------------------------------------------------------------------- #
def _prepare_config(model_dir: str, n_dmrs, loss: str):
    """Load a MethylBERT config and force ``num_labels`` cleanly.

    Passing ``num_labels`` directly to ``from_pretrained`` conflicts with the
    pretrained config's 2-entry ``id2label`` map (HuggingFace then resets
    ``num_labels`` to -1, which breaks the DMR embedding). Setting the property
    here regenerates ``id2label`` consistently, avoiding the conflict.
    """
    from methylbert.config import MethylBERTConfig

    config = MethylBERTConfig.from_pretrained(model_dir)
    if n_dmrs is not None:
        n = int(n_dmrs)
        # MethylBERTConfig shadows num_labels as a plain attribute, so the
        # property setter does not rebuild id2label. Keep both in sync so the
        # value round-trips through save_pretrained / from_pretrained.
        config.num_labels = n
        config.id2label = {i: f"LABEL_{i}" for i in range(n)}
        config.label2id = {f"LABEL_{i}": i for i in range(n)}
    config.loss = loss
    config.output_attentions = False
    config.output_hidden_states = False
    return config


def build_pretrained_model(pretrained_dir: str, n_dmrs: int,
                           seq_len: int = DEFAULT_SEQ_LEN, loss: str = "bce"):
    """Initialise a read-classification model from a pretrained MethylBERT MLM."""
    from methylbert.network import MethylBertEmbeddedDMR

    config = _prepare_config(pretrained_dir, n_dmrs, loss)
    return MethylBertEmbeddedDMR.from_pretrained(
        pretrained_dir, config=config, seq_len=seq_len)


def _infer_n_dmrs_from_dir(model_dir: str):
    """Read the DMR embedding size from a saved checkpoint's weights."""
    safetensors = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(safetensors):
        from safetensors import safe_open

        with safe_open(safetensors, framework="pt") as f:
            if "dmr_encoder.0.weight" in f.keys():
                return int(f.get_slice("dmr_encoder.0.weight").get_shape()[0])
    pickle = os.path.join(model_dir, "dmr_encoder.pickle")
    if os.path.exists(pickle):
        import torch

        sd = torch.load(pickle, map_location="cpu")
        return int(sd["0.weight"].shape[0])
    return None


def load_finetuned_model(model_dir: str, seq_len: int = DEFAULT_SEQ_LEN,
                         loss: str = "bce", n_dmrs=None):
    """Load a fine-tuned model directory written by :func:`save_finetuned_model`."""
    from methylbert.network import MethylBertEmbeddedDMR

    if n_dmrs is None:
        n_dmrs = _infer_n_dmrs_from_dir(model_dir)
    if n_dmrs is None:
        raise ValueError(
            "Could not determine n_dmrs from the model directory; "
            "pass --dmr / n_dmrs explicitly.")
    config = _prepare_config(model_dir, n_dmrs, loss)
    return MethylBertEmbeddedDMR.from_pretrained(
        model_dir, config=config, seq_len=seq_len)


def save_finetuned_model(model, out_dir: str):
    """Save model weights/config plus submodule pickles *inside* ``out_dir``.

    The live model is *not* moved across devices: under DDP, moving the wrapped
    module to CPU and back reallocates its parameter storages and invalidates the
    reducer's gradient buckets, which deadlocks the next collective. We instead
    serialize CPU copies of the state dicts and let ``save_pretrained`` handle the
    device transfer internally.
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)
    m = model.module if hasattr(model, "module") else model
    # Serialize a CPU copy so safetensors is happy and the live (GPU/DDP) model
    # is never moved or mutated.
    cpu_state = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    m.save_pretrained(out_dir, state_dict=cpu_state)
    if hasattr(m, "read_classifier"):
        torch.save({k: v.detach().cpu() for k, v in m.read_classifier.state_dict().items()},
                   os.path.join(out_dir, "read_classification_model.pickle"))
    if hasattr(m, "dmr_encoder"):
        torch.save({k: v.detach().cpu() for k, v in m.dmr_encoder.state_dict().items()},
                   os.path.join(out_dir, "dmr_encoder.pickle"))


# --------------------------------------------------------------------------- #
# Precision helpers
# --------------------------------------------------------------------------- #
def resolve_amp(precision: str):
    """Return (autocast_enabled, torch_dtype, use_grad_scaler) for a precision str."""
    import torch

    precision = (precision or "fp32").lower()
    if precision in ("fp16", "float16", "half"):
        return True, torch.float16, True
    if precision in ("bf16", "bfloat16"):
        return True, torch.bfloat16, False
    return False, torch.float32, False


# --------------------------------------------------------------------------- #
# Read-level inference + metrics
# --------------------------------------------------------------------------- #
def run_read_level_inference(model, data_loader, device, precision="fp32",
                             with_labels=True, progress=False):
    """Run the model forward pass over a loader and return per-read probabilities.

    Returns a dict of numpy arrays: name, prob_class_0, prob_class_1, pred,
    dmr_label and (optionally) ctype_label. This reuses the official
    ``MethylBertEmbeddedDMR.forward`` (which already softmaxes the read
    classification logits) but never performs sample-level deconvolution.
    """
    import torch

    amp_enabled, amp_dtype, _ = resolve_amp(precision)
    model.eval()

    names, p1, preds, dmrs, labels = [], [], [], [], []
    iterator = data_loader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(data_loader, desc="read-level inference")
        except Exception:
            iterator = data_loader

    with torch.no_grad():
        for batch in iterator:
            ids = batch["dna_seq"].to(device, non_blocking=True)
            tt = batch["methyl_seq"].to(device, non_blocking=True)
            dmr = batch["dmr_label"].to(device, non_blocking=True)
            ctype = batch.get("ctype_label")
            ctype_dev = (ctype.to(device, non_blocking=True)
                         if ctype is not None else torch.zeros_like(dmr))
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=amp_dtype, enabled=amp_enabled):
                out = model(step=0, input_ids=ids, token_type_ids=tt,
                            labels=dmr, ctype_label=ctype_dev)
            probs = out["classification_logits"].float().cpu().numpy()
            p1.append(probs[:, 1])
            preds.append(probs.argmax(axis=1))
            dmrs.append(batch["dmr_label"].numpy())
            names.extend(batch["name"])
            if with_labels and ctype is not None:
                labels.append(batch["ctype_label"].numpy())

    res = {
        "name": np.asarray(names),
        "prob_class_1": np.concatenate(p1) if p1 else np.array([]),
        "pred": np.concatenate(preds) if preds else np.array([], dtype=int),
        "dmr_label": np.concatenate(dmrs) if dmrs else np.array([], dtype=int),
    }
    res["prob_class_0"] = 1.0 - res["prob_class_1"]
    if with_labels and labels:
        res["ctype_label"] = np.concatenate(labels)
    return res


def classification_metrics(y_true, y_prob, y_pred):
    """AUC / PR-AUC / balanced accuracy / accuracy with robust degenerate handling."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                  balanced_accuracy_score, roc_auc_score)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)
    out = {
        "n": int(y_true.shape[0]),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "auc": float("nan"),
        "pr_auc": float("nan"),
    }
    if y_true.shape[0] == 0:
        return out
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    try:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        pass
    if len(np.unique(y_true)) > 1:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass
    return out


def count_cpg_from_token_type(token_type_ids) -> int:
    """Number of CpG positions in a token_type (methyl) array: values 0 or 1."""
    arr = np.asarray(token_type_ids)
    return int(np.count_nonzero(arr <= 1))
