from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import gradio as gr

from main import build_dataset


def run_with_progress(url: str, output_path: str):
    if not url.strip():
        yield "YouTube URL を入力してください。", ""
        return

    out = output_path.strip() or "output"

    log_q: queue.Queue[str] = queue.Queue()
    done = threading.Event()
    result: dict[str, str] = {"zip_path": "", "error": ""}

    def log(message: str) -> None:
        log_q.put(message)

    def worker() -> None:
        try:
            zip_path = build_dataset(url.strip(), Path(out), progress=log, require_gpu=True)
            result["zip_path"] = str(zip_path)
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    lines: list[str] = []
    while not done.is_set() or not log_q.empty():
        try:
            while True:
                lines.append(log_q.get_nowait())
        except queue.Empty:
            pass

        status = "\n".join(lines) if lines else "処理を開始しました..."
        yield status, ""
        time.sleep(0.5)

    if result["error"]:
        final_log = "\n".join(lines + [f"\nエラー: {result['error']}"])
        yield final_log, ""
        return

    final_log = "\n".join(lines)
    yield final_log, result["zip_path"]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RVC Dataset Maker (Colab GPU)") as demo:
        gr.Markdown("# RVC Dataset Maker (Colab GPU)")
        gr.Markdown("YouTube URL から音声を抽出し、BGM を除去して wav 分割 + zip 化します。")

        url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        output = gr.Textbox(label="Output Path", value="/content/output")
        run_button = gr.Button("実行")

        log_box = gr.Textbox(label="進捗ログ", lines=18)
        zip_path = gr.Textbox(label="生成ZIPのパス")

        run_button.click(
            fn=run_with_progress,
            inputs=[url, output],
            outputs=[log_box, zip_path],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RVC Dataset Maker Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio の公開リンクを有効化")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
