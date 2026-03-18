from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List

import librosa
import numpy as np
import soundfile as sf
import torch
from yt_dlp import YoutubeDL

from slicer2 import Slicer


TARGET_SR = 40000

# slicer2 settings
SLICER_THRESHOLD_DB = -40
SLICER_MIN_LENGTH_MS = 5000
SLICER_MIN_INTERVAL_MS = 300
SLICER_HOP_SIZE_MS = 10
SLICER_MAX_SIL_KEPT_MS = 500

# clip filters
MIN_CLIP_SEC = 2.0
MAX_CLIP_SEC = 15.0
MIN_RMS_DB = -38.0
MAX_RMS_DB = -3.0
MAX_PEAK_ABS = 0.999
MAX_CLIPPING_RATIO = 0.01
MIN_VOICE_RATIO = 0.35
MAX_ZCR = 0.25
MIN_SPECTRAL_CENTROID = 80.0
MAX_SPECTRAL_CENTROID = 7500.0
PEAK_NORM = 0.98

# ffmpeg preprocess
HPF_FREQ = 70
LPF_FREQ = 16000

# demucs
DEMUCS_MODEL = "htdemucs"
DEMUCS_TWO_STEMS = "vocals"

ProgressFn = Callable[[str], None]


@dataclass
class SegmentMeta:
    file: str
    duration_sec: float
    rms_db: float
    zcr: float
    spectral_centroid: float
    clipping_ratio: float


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:max_len] or "dataset"


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg が見つかりません。PATH に追加してください。")


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU (CUDA) が利用できません。Colab ではランタイムを GPU に変更して再実行してください。"
        )


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        cwd=str(cwd) if cwd else None,
    )

    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(
            "コマンド実行に失敗しました。\n"
            f"cmd: {' '.join(cmd)}\n\n"
            f"stdout:\n{stdout}\n\n"
            f"stderr:\n{stderr}"
        )


def dbfs_rms(y: np.ndarray) -> float:
    eps = 1e-10
    rms = np.sqrt(np.mean(np.square(y), dtype=np.float64) + eps)
    return 20.0 * math.log10(rms + eps)


def peak_normalize(y: np.ndarray, peak: float = PEAK_NORM) -> np.ndarray:
    if len(y) == 0:
        return y
    max_abs = np.max(np.abs(y))
    if max_abs < 1e-9:
        return y
    return (y / max_abs) * peak


def split_if_too_long(y: np.ndarray, sr: int, max_sec: float = MAX_CLIP_SEC) -> List[np.ndarray]:
    max_len = int(sr * max_sec)
    if len(y) <= max_len:
        return [y]

    chunks: List[np.ndarray] = []
    cursor = 0
    while cursor < len(y):
        end = min(cursor + max_len, len(y))
        chunks.append(y[cursor:end])
        cursor = end
    return chunks


def download_audio(url: str, out_dir: Path) -> tuple[Path, str]:
    outtmpl = str(out_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))
        title = info.get("title", "dataset")

    return downloaded_path, title


def convert_to_clean_wav(src_path: Path, dst_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-af",
        f"highpass=f={HPF_FREQ},lowpass=f={LPF_FREQ}",
        str(dst_path),
    ]
    run_cmd(cmd)


def run_demucs(input_wav: Path, demucs_out_dir: Path, device: str = "cuda") -> Path:
    demucs_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        DEMUCS_MODEL,
        "--two-stems",
        DEMUCS_TWO_STEMS,
        "-d",
        device,
        "-o",
        str(demucs_out_dir),
        str(input_wav),
    ]
    run_cmd(cmd)

    expected = demucs_out_dir / DEMUCS_MODEL / input_wav.stem / "vocals.wav"
    if not expected.exists():
        raise RuntimeError(f"Demucs の出力が見つかりません: {expected}")

    return expected


def clipping_ratio(y: np.ndarray, threshold: float = 0.99) -> float:
    if len(y) == 0:
        return 1.0
    return float(np.mean(np.abs(y) >= threshold))


def zero_crossing_rate_value(y: np.ndarray) -> float:
    if len(y) < 2:
        return 1.0
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)
    return float(np.mean(zcr))


def spectral_centroid_value(y: np.ndarray, sr: int) -> float:
    if len(y) == 0:
        return 0.0
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
    return float(np.mean(centroid))


def voice_active_ratio(y: np.ndarray, sr: int, top_db: float = 35.0) -> float:
    if len(y) == 0:
        return 0.0
    intervals = librosa.effects.split(y, top_db=top_db)
    voiced = sum(int(end - start) for start, end in intervals)
    return voiced / max(len(y), 1)


def is_garbage_clip(y: np.ndarray, sr: int) -> tuple[bool, dict]:
    duration_sec = len(y) / sr
    rms_db = dbfs_rms(y)
    zcr = zero_crossing_rate_value(y)
    centroid = spectral_centroid_value(y, sr)
    clip_ratio = clipping_ratio(y)
    voice_ratio = voice_active_ratio(y, sr)

    reasons = {
        "duration_sec": duration_sec,
        "rms_db": rms_db,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "clipping_ratio": clip_ratio,
        "voice_ratio": voice_ratio,
    }

    if duration_sec < MIN_CLIP_SEC:
        return True, reasons
    if rms_db < MIN_RMS_DB:
        return True, reasons
    if rms_db > MAX_RMS_DB:
        return True, reasons
    if clip_ratio > MAX_CLIPPING_RATIO:
        return True, reasons
    if voice_ratio < MIN_VOICE_RATIO:
        return True, reasons
    if zcr > MAX_ZCR:
        return True, reasons
    if centroid < MIN_SPECTRAL_CENTROID or centroid > MAX_SPECTRAL_CENTROID:
        return True, reasons
    if np.max(np.abs(y)) > MAX_PEAK_ABS:
        return True, reasons

    return False, reasons


def extract_segments_with_slicer2(wav_path: Path, raw_dir: Path, rejected_dir: Path) -> List[SegmentMeta]:
    audio, sr = librosa.load(wav_path, sr=None, mono=False)

    slicer = Slicer(
        sr=sr,
        threshold=SLICER_THRESHOLD_DB,
        min_length=SLICER_MIN_LENGTH_MS,
        min_interval=SLICER_MIN_INTERVAL_MS,
        hop_size=SLICER_HOP_SIZE_MS,
        max_sil_kept=SLICER_MAX_SIL_KEPT_MS,
    )

    raw_chunks = slicer.slice(audio)
    metas: List[SegmentMeta] = []
    seg_idx = 1
    rej_idx = 1

    for chunk in raw_chunks:
        if len(chunk.shape) > 1:
            if chunk.shape[0] <= 8:
                chunk = chunk.mean(axis=0)
            else:
                chunk = chunk.mean(axis=1)

        sub_chunks = split_if_too_long(chunk, sr, max_sec=MAX_CLIP_SEC)

        for sub_chunk in sub_chunks:
            reject, stats = is_garbage_clip(sub_chunk, sr)

            if reject:
                rejected_path = rejected_dir / f"rej_{rej_idx:05d}.wav"
                sf.write(rejected_path, sub_chunk, sr)
                rej_idx += 1
                continue

            sub_chunk = peak_normalize(sub_chunk)

            filename = f"{seg_idx:05d}.wav"
            out_path = raw_dir / filename
            sf.write(out_path, sub_chunk, sr)

            metas.append(
                SegmentMeta(
                    file=filename,
                    duration_sec=round(stats["duration_sec"], 3),
                    rms_db=round(stats["rms_db"], 2),
                    zcr=round(stats["zcr"], 4),
                    spectral_centroid=round(stats["spectral_centroid"], 2),
                    clipping_ratio=round(stats["clipping_ratio"], 6),
                )
            )
            seg_idx += 1

    return metas


def make_readme(dataset_name: str, source_url: str, metas: List[SegmentMeta]) -> str:
    total_sec = sum(m.duration_sec for m in metas)
    return (
        f"# {dataset_name}\n\n"
        f"Source URL:\n{source_url}\n\n"
        f"Files:\n{len(metas)}\n\n"
        f"Total duration (sec):\n{round(total_sec, 2)}\n\n"
        f"Sample rate:\n{TARGET_SR}\n\n"
        "Pipeline:\n"
        "- yt-dlp download\n"
        "- ffmpeg preprocess\n"
        "- Demucs vocals separation\n"
        "- slicer2 silence slicing\n"
        "- garbage clip filtering\n"
        "- zip packaging\n\n"
        "Slicer settings:\n"
        f"- threshold = {SLICER_THRESHOLD_DB}\n"
        f"- min_length = {SLICER_MIN_LENGTH_MS}\n"
        f"- min_interval = {SLICER_MIN_INTERVAL_MS}\n"
        f"- hop_size = {SLICER_HOP_SIZE_MS}\n"
        f"- max_sil_kept = {SLICER_MAX_SIL_KEPT_MS}\n\n"
        "Garbage filter settings:\n"
        f"- min_clip_sec = {MIN_CLIP_SEC}\n"
        f"- max_clip_sec = {MAX_CLIP_SEC}\n"
        f"- min_rms_db = {MIN_RMS_DB}\n"
        f"- max_rms_db = {MAX_RMS_DB}\n"
        f"- max_clipping_ratio = {MAX_CLIPPING_RATIO}\n"
        f"- min_voice_ratio = {MIN_VOICE_RATIO}\n"
        f"- max_zcr = {MAX_ZCR}\n"
        f"- centroid_range = [{MIN_SPECTRAL_CENTROID}, {MAX_SPECTRAL_CENTROID}]\n"
    )


def zip_raw_directory(raw_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for wav in raw_dir.glob("*.wav"):
            zf.write(wav, arcname=f"raw/{wav.name}")


def build_dataset(
    url: str,
    output_root: Path,
    progress: ProgressFn | None = None,
    require_gpu: bool = True,
) -> Path:
    def elapsed_sec(started_at: float) -> str:
        return f"{time.perf_counter() - started_at:.2f}s"

    def log(message: str) -> None:
        if progress is not None:
            progress(message)
        else:
            print(message)

    check_ffmpeg()

    if require_gpu:
        ensure_cuda_available()
        gpu_name = torch.cuda.get_device_name(0)
        log(f"[GPU] torch={torch.__version__} cuda={torch.version.cuda} device={gpu_name}")

    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        download_dir = tmp_dir / "download"
        work_dir = tmp_dir / "work"
        demucs_dir = tmp_dir / "demucs"

        download_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        demucs_dir.mkdir(parents=True, exist_ok=True)

        log("[1/6] YouTube から音声を取得中...")
        t1 = time.perf_counter()
        downloaded_path, title = download_audio(url, download_dir)
        dataset_name = slugify(title)
        log(f"[1/6] 完了 ({elapsed_sec(t1)})")

        clean_wav = work_dir / "clean.wav"

        log("[2/6] WAV に変換中...")
        t2 = time.perf_counter()
        convert_to_clean_wav(downloaded_path, clean_wav)
        log(f"[2/6] 完了 ({elapsed_sec(t2)})")

        log("[3/6] Demucs で BGM 除去中...")
        t3 = time.perf_counter()
        vocals_wav = run_demucs(clean_wav, demucs_dir, device="cuda" if require_gpu else "cpu")
        log(f"[3/6] 完了 ({elapsed_sec(t3)})")

        dataset_dir = output_root / dataset_name
        raw_dir = dataset_dir / "raw"
        rejected_dir = dataset_dir / "rejected"

        dataset_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)

        log("[4/6] slicer2 で分割 + ゴミ除去中...")
        t4 = time.perf_counter()
        metas = extract_segments_with_slicer2(vocals_wav, raw_dir, rejected_dir)
        log(f"[4/6] 完了 ({elapsed_sec(t4)})")

        if not metas:
            raise RuntimeError(
                "有効なセグメントを抽出できませんでした。"
                "しきい値が厳しすぎるか、音声に問題がある可能性があります。"
            )

        log("[5/6] metadata / README を出力中...")
        t5 = time.perf_counter()
        meta_path = dataset_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": dataset_name,
                    "source_url": url,
                    "sample_rate": TARGET_SR,
                    "segment_count": len(metas),
                    "segments": [asdict(m) for m in metas],
                    "slicer": {
                        "type": "slicer2",
                        "threshold": SLICER_THRESHOLD_DB,
                        "min_length": SLICER_MIN_LENGTH_MS,
                        "min_interval": SLICER_MIN_INTERVAL_MS,
                        "hop_size": SLICER_HOP_SIZE_MS,
                        "max_sil_kept": SLICER_MAX_SIL_KEPT_MS,
                    },
                    "filter": {
                        "min_clip_sec": MIN_CLIP_SEC,
                        "max_clip_sec": MAX_CLIP_SEC,
                        "min_rms_db": MIN_RMS_DB,
                        "max_rms_db": MAX_RMS_DB,
                        "max_clipping_ratio": MAX_CLIPPING_RATIO,
                        "min_voice_ratio": MIN_VOICE_RATIO,
                        "max_zcr": MAX_ZCR,
                        "min_spectral_centroid": MIN_SPECTRAL_CENTROID,
                        "max_spectral_centroid": MAX_SPECTRAL_CENTROID,
                    },
                    "demucs": {
                        "model": DEMUCS_MODEL,
                        "two_stems": DEMUCS_TWO_STEMS,
                        "device": "cuda" if require_gpu else "cpu",
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        readme_path = dataset_dir / "README.txt"
        readme_path.write_text(make_readme(dataset_name, url, metas), encoding="utf-8")
        log(f"[5/6] 完了 ({elapsed_sec(t5)})")

        log("[6/6] ZIP 作成中...")
        t6 = time.perf_counter()
        zip_path = dataset_dir / f"{dataset_name}.zip"
        zip_raw_directory(raw_dir, zip_path)
        log(f"[6/6] 完了 ({elapsed_sec(t6)})")

        log(f"完了: {zip_path}")
        return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube URL から RVC 用データセット ZIP を生成する")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument(
        "--output",
        default="output",
        help="出力ディレクトリ (default: output)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="GPU がない環境でも CPU で実行する (Colab では通常不要)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        zip_path = build_dataset(args.url, Path(args.output), require_gpu=not args.allow_cpu)
        print(f"\n完了: {zip_path}")
        return 0
    except Exception as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
