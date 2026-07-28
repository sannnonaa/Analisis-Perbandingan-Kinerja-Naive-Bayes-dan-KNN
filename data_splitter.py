from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────── #
#  ENUM
# ─────────────────────────────────────────────────────────────────────────── #

class SplitMode(Enum):
    AUTO   = auto()   # sklearn train_test_split berdasarkan rasio
    MANUAL = auto()   # pengguna menentukan file test secara eksplisit


# ─────────────────────────────────────────────────────────────────────────── #
#  CONFIG & RESULT
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class SplitConfig:
    mode:                  SplitMode  = SplitMode.AUTO
    test_size:             float      = 0.20   # AUTO only
    random_state:          int        = 42     # AUTO only
    stratify:              bool       = True   # AUTO only
    manual_test_filenames: set[str]   = field(default_factory=set)  # MANUAL only


@dataclass
class SplitResult:
    train_idx:      np.ndarray
    test_idx:       np.ndarray
    y_train:        np.ndarray
    y_test:         np.ndarray
    test_filenames: list[str]

    n_total:     int   = 0
    n_train:     int   = 0
    n_test:      int   = 0
    ratio_train: float = 0.0
    ratio_test:  float = 0.0
    class_dist:  dict  = field(default_factory=dict)

    def __post_init__(self):
        self.n_train     = len(self.train_idx)
        self.n_test      = len(self.test_idx)
        self.n_total     = self.n_train + self.n_test
        self.ratio_train = self.n_train / self.n_total * 100 if self.n_total else 0.0
        self.ratio_test  = self.n_test  / self.n_total * 100 if self.n_total else 0.0


# ─────────────────────────────────────────────────────────────────────────── #
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────── #

def run_split(df: pd.DataFrame, config: SplitConfig) -> SplitResult:
    _validate(df)
    if config.mode == SplitMode.MANUAL:
        return _run_manual_split(df, config)
    return _run_auto_split(df, config)


# ─────────────────────────────────────────────────────────────────────────── #
#  AUTO SPLIT
# ─────────────────────────────────────────────────────────────────────────── #

def _run_auto_split(df: pd.DataFrame, config: SplitConfig) -> SplitResult:
    labels = df["label"].values
    n      = len(df)

    stratify_arr = labels if config.stratify else None

    try:
        tr_idx, ts_idx = train_test_split(
            np.arange(n),
            test_size    = config.test_size,
            random_state = config.random_state,
            stratify     = stratify_arr,
        )
    except ValueError as e:
        if config.stratify:
            raise RuntimeError(
                f"Stratify gagal: {e}\n\n"
                "Coba nonaktifkan opsi Stratify, atau tambah jumlah sampel per kelas."
            ) from e
        raise

    result = SplitResult(
        train_idx      = tr_idx,
        test_idx       = ts_idx,
        y_train        = labels[tr_idx],
        y_test         = labels[ts_idx],
        test_filenames = df.iloc[ts_idx]["filename"].tolist(),
    )
    result.class_dist = _build_class_dist(labels, tr_idx, ts_idx)
    return result


# ─────────────────────────────────────────────────────────────────────────── #
#  MANUAL SPLIT
# ─────────────────────────────────────────────────────────────────────────── #

def _run_manual_split(df: pd.DataFrame, config: SplitConfig) -> SplitResult:
    selected = config.manual_test_filenames

    if not selected:
        raise ValueError(
            "Mode Manual: belum ada file yang dipilih sebagai Test set.\n"
            "Pilih minimal 1 file dari panel pemilihan file."
        )

    filenames = df["filename"].tolist()

    not_found = selected - set(filenames)
    if not_found:
        raise ValueError(
            "File berikut tidak ditemukan di dataset:\n"
            + "\n".join(f"  • {f}" for f in sorted(not_found))
        )

    ts_idx = np.array([i for i, fn in enumerate(filenames) if fn in selected])
    tr_idx = np.array([i for i, fn in enumerate(filenames) if fn not in selected])

    if len(tr_idx) == 0:
        raise ValueError(
            "Semua dokumen dipilih sebagai Test set — Train set menjadi kosong.\n"
            "Lepas pilihan setidaknya 1 dokumen agar Train set tidak kosong."
        )

    labels = df["label"].values
    result = SplitResult(
        train_idx      = tr_idx,
        test_idx       = ts_idx,
        y_train        = labels[tr_idx],
        y_test         = labels[ts_idx],
        test_filenames = df.iloc[ts_idx]["filename"].tolist(),
    )
    result.class_dist = _build_class_dist(labels, tr_idx, ts_idx)
    return result


# ─────────────────────────────────────────────────────────────────────────── #
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────── #

def _validate(df: pd.DataFrame) -> None:
    for col in ("label", "text_clean", "filename"):
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di DataFrame.")
    if len(df) < 4:
        raise ValueError("Dataset terlalu kecil untuk dibagi (minimal 4 baris).")


def _build_class_dist(
    labels:  np.ndarray,
    tr_idx:  np.ndarray,
    ts_idx:  np.ndarray,
) -> dict[str, dict[str, int]]:
    train_labels = labels[tr_idx]
    test_labels  = labels[ts_idx]
    dist = {}
    for lbl in sorted(set(labels)):
        dist[lbl] = {
            "train": int((train_labels == lbl).sum()),
            "test":  int((test_labels  == lbl).sum()),
        }
    return dist


def save_to_controller(controller, result: SplitResult) -> None:
    controller.train_idx      = result.train_idx
    controller.test_idx       = result.test_idx
    controller.y_train        = result.y_train
    controller.y_test         = result.y_test
    controller.test_filenames = result.test_filenames


def validate_train_test_labels(y_train: np.ndarray, y_test: np.ndarray) -> set:
    train_labels = set(np.unique(y_train))
    test_labels  = set(np.unique(y_test))
    return test_labels - train_labels