"""Loading and validating the UCI Parkinson's Disease Classification dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


UCI_DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification"
)
# UCI distributes the CSV inside a RAR archive. This CSV mirror makes the
# automatic download cross-platform; its dimensions are validated below.
CSV_MIRROR_URL = (
    "https://huggingface.co/datasets/wwydmanski/ParkinsonsDisease/resolve/"
    "main/pd_speech_features.csv?download=true"
)
DEFAULT_CACHE_PATH = Path("data/pd_speech_features.csv")


@dataclass(frozen=True)
class VoiceDataset:
    """Acoustic features, labels, subject IDs, and audit-only demographics."""

    features: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    demographics: pd.DataFrame
    source: str


def _read_uci470_csv(path: Path) -> pd.DataFrame:
    """Read either the original two-header CSV or an already cleaned export."""

    frame = pd.read_csv(path, skiprows=1)
    if {"id", "gender", "class"}.issubset(frame.columns):
        return frame
    frame = pd.read_csv(path)
    if {"id", "gender", "class"}.issubset(frame.columns):
        return frame
    raise ValueError(
        "Expected UCI 470 columns 'id', 'gender', and 'class'. The older UCI "
        "174 file is not compatible with this uncertainty study."
    )


def _from_frame(frame: pd.DataFrame, source: str) -> VoiceDataset:
    required = {"id", "gender", "class"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {', '.join(sorted(missing))}"
        )

    groups = pd.to_numeric(frame["id"], errors="raise").astype(int).rename("subject_id")
    target = pd.to_numeric(frame["class"], errors="raise").astype(int).rename("class")
    gender = pd.to_numeric(frame["gender"], errors="raise").astype(int).rename("gender")
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("The class target must contain only 0 (healthy) and 1 (PD).")
    if not set(gender.unique()).issubset({0, 1}):
        raise ValueError("The gender field must contain only 0 (female) and 1 (male).")

    # Gender is deliberately withheld from model inputs. It is an audit
    # variable, not an acoustic voice marker.
    features = frame.drop(columns=["id", "gender", "class"]).apply(
        pd.to_numeric, errors="raise"
    )
    if features.empty:
        raise ValueError("The dataset does not contain acoustic predictors.")
    if features.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate feature names.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Acoustic predictors contain missing or infinite values.")

    subject_frame = pd.DataFrame(
        {"subject_id": groups, "class": target, "gender": gender}
    )
    consistency = subject_frame.groupby("subject_id")[["class", "gender"]].nunique()
    if (consistency > 1).any().any():
        raise ValueError("At least one subject has inconsistent diagnosis or gender.")

    demographics = pd.DataFrame(
        {"gender": gender, "sex": gender.map({0: "Female", 1: "Male"})}
    )
    return VoiceDataset(
        features=features.reset_index(drop=True),
        target=target.reset_index(drop=True),
        groups=groups.reset_index(drop=True),
        demographics=demographics.reset_index(drop=True),
        source=source,
    )


def _download_csv(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(CSV_MIRROR_URL, headers={"User-Agent": "Mozilla/5.0"})
    temporary = destination.with_suffix(".download")
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            output.write(response.read())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_voice_dataset(
    csv_path: str | Path | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> VoiceDataset:
    """Load UCI dataset 470 locally, downloading and caching it when needed."""

    if csv_path is not None:
        path = Path(csv_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return _from_frame(_read_uci470_csv(path), str(path))

    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        try:
            _download_csv(path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the UCI 470 CSV mirror. Download "
                f"pd_speech_features.csv from {UCI_DATASET_PAGE} and pass --data."
            ) from exc
    dataset = _from_frame(
        _read_uci470_csv(path),
        f"UCI Parkinson's Disease Classification dataset 470 (cached at {path})",
    )
    if len(dataset.target) != 756 or dataset.groups.nunique() != 252:
        raise ValueError(
            "The cached automatic-download dataset failed validation: expected "
            "756 recordings from 252 subjects."
        )
    return dataset
