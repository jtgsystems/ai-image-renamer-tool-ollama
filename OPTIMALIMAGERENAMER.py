#!/usr/bin/env python3
"""
OPTIMAL IMAGE RENAMER
=====================
A **one-file, drop-in** tool that renames images *in-place* using
Ollama's LLaVA vision-language model.  It tries to combine the best
ideas that are scattered across all previous renamers that live in the
`llava-image-renamer` folder:

* Safe in-place renaming with automatic conflict resolution.
* Duplicate detection via MD5 hash so the same image is never processed
  twice in a single run.
* High-quality preprocessing – we **never** overwrite or recompress the
  original file, we only convert to RGB in memory when the model
  requires it.
* Sensible defaults that balance quality and speed (temperature 0.2,
  top-p 0.8, top-k 20, repeat-penalty 1.05).
* Multithreading that automatically scales to the number of CPU cores,
  but can be overridden with `--workers`.

The result is a script that should “just work” on any machine that has
Python ≥3.8, Ollama and the *llava* model installed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import subprocess
import atexit

# ---------------------------------------------------------------------------
# Third-party imports – we defer them so we can install missing packages
# automatically on first run.
# ---------------------------------------------------------------------------

try:
    import ollama  # type: ignore
    from PIL import Image  # type: ignore
except ModuleNotFoundError as e:
    print(f"⚠️  Missing dependency: {e.name}. Installing… (this can take a moment)")
    import subprocess

    pkgs = ["ollama", "pillow"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])

    import ollama  # type: ignore  # noqa: E402
    from PIL import Image  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Helper constants / functions
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
    ".heic", ".heif", ".avif", ".ico", ".raw", ".cr2", ".nef",
}


def is_image_file(path: Path) -> bool:
    """Return *True* if *path* looks like a real image file."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def md5sum(path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# ---------------------------------------------------------------------------
# Core renamer class
# ---------------------------------------------------------------------------


class OptimalImageRenamer:
    """High-level façade that does all the heavy lifting."""

    # Ollama generation parameters chosen after benchmarking the various
    # existing renamers.
    OLLAMA_OPTIONS = {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "repeat_penalty": 1.05,
        "num_predict": 60,
    }

    PROMPT = (
        "Describe this image in **exactly 12 words**. "
        "Focus on the main subject, important objects and the setting. "
        "Be specific, use descriptive adjectives, avoid redundancy."
    )

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "this", "that", "these", "those", "there", "where",
        "when", "how", "what", "which",
    }

    def __init__(self, model: str = "llava:latest", target_words: int = 12, *, max_workers: int | None = None):
        self.model = model
        self.target_words = target_words
        self.max_workers = max_workers or max(os.cpu_count() or 4, 4)

        # Detect GPUs and enable Ollama scheduling across them.
        self.gpu_count = self._detect_gpu_count()
        if self.gpu_count > 1 and os.getenv("OLLAMA_SCHED_SPREAD") != "1":
            # Tell Ollama server (that we may launch in a moment) to spread across GPUs.
            os.environ["OLLAMA_SCHED_SPREAD"] = "1"

        if self.gpu_count:
            print(f"🖥️  Detected {self.gpu_count} CUDA GPU(s)")
        else:
            print("🖥️  No CUDA GPUs detected – will run on CPU (much slower)")

        # Bookkeeping
        self._hashes_seen: set[str] = set()
        self._results: List[Dict] = []

        # Local server process handle (if we spawn one)
        self._ollama_proc: Optional[subprocess.Popen] = None

        atexit.register(self._cleanup_server)

        # Quick connectivity test – if it fails we try to spin up a local server.
        self._ensure_ollama_server_ready()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def process(self, directory: Path, *, auto_confirm: bool = False) -> None:
        images = self._gather_images(directory)
        if not images:
            print("❌ No images found.")
            return

        if len(images) > 50 and not auto_confirm:
            print(f"⚠️  About to rename {len(images)} files in-place. Backups are *not* created.")
            if input("Proceed? (y/N): ").strip().lower() not in {"y", "yes"}:
                print("🚪 Aborted by user.")
                return

        print(f"🚀 Starting rename of {len(images)} images using up to {self.max_workers} threads…\n")

        start = time.time()
        total = len(images)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_one, img): img for img in images}
            for idx, future in enumerate(as_completed(futures), 1):
                res = future.result()
                self._results.append(res)

                status = "✅" if res["success"] else "❌"
                old_name = Path(res["original"]).name
                new_name = Path(res.get("new_name", "")).name if res.get("new_name") else old_name
                changed_indicator = "→ " + new_name if new_name != old_name else "(unchanged)"
                if not res["success"]:
                    changed_indicator = f"✖ {res.get('error', 'error')}"[:60]

                progress = idx / total * 100
                elapsed = time.time() - start
                speed = idx / elapsed if elapsed > 0 else 0

                print(f"{status} {idx:4d}/{total} {progress:6.2f}% {speed:5.1f} img/s  {old_name}  {changed_indicator}", flush=True)

        elapsed = time.time() - start
        self._print_summary(elapsed)

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def _cleanup_server(self) -> None:
        if getattr(self, "_ollama_proc", None):
            try:
                self._ollama_proc.terminate()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # GPU & Ollama server helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_gpu_count() -> int:
        """Return how many NVIDIA GPUs are present (best-effort)."""
        try:
            output = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True)
            return len([ln for ln in output.splitlines() if "GPU" in ln])
        except Exception:
            return 0

    def _ensure_ollama_server_ready(self) -> None:
        """Ping Ollama, start it locally if it is not running."""
        try:
            self._test_ollama()
            return  # already running
        except SystemExit:
            raise  # _test_ollama already printed & exited
        except Exception:
            pass  # will attempt to start below

        print("🔄 Starting local Ollama server…")
        # Start server in the background; inherit env so GPU spread flag is respected.
        self._ollama_proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Give the server a bit of time to start up.
        for _ in range(20):
            try:
                time.sleep(0.5)
                self._test_ollama(silent=True)
                print("✅ Ollama server is up.")
                return
            except Exception:
                continue

        print("❌ Failed to start Ollama server automatically. Please start it manually (\n    ollama serve\n) and retry.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Internals – single image pipeline
    # ------------------------------------------------------------------

    def _process_one(self, path: Path) -> Dict:
        out: Dict[str, object] = {
            "original": str(path),
            "success": False,
            "new_name": None,
            "description": None,
            "error": None,
        }

        try:
            file_hash = md5sum(path)
            if file_hash in self._hashes_seen:
                out["error"] = "duplicate-image"
                return out
            self._hashes_seen.add(file_hash)

            # Load + minimally preprocess image (never touches disk).
            image = self._load_image(path)

            # Ask the model.
            desc = self._describe(image)
            if not desc:
                out["error"] = "no-description"
                return out

            filename_safe = self._to_filename(desc)
            new_path = path.with_name(f"{filename_safe}{path.suffix.lower()}")

            # Resolve collisions.
            counter = 1
            while new_path.exists() and new_path != path:
                new_path = path.with_name(f"{filename_safe}-{counter:02d}{path.suffix.lower()}")
                counter += 1

            if new_path != path:
                os.rename(path, new_path)

            out.update(success=True, new_name=str(new_path), description=desc)
            return out

        except Exception as e:  # pylint: disable=broad-except
            out["error"] = str(e)
            return out

    # ------------------------------------------------------------------
    # Internals – helper methods
    # ------------------------------------------------------------------

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            max_side = 4096  # Enough for detail, avoids >8K overkill.
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

            return img.copy()

    def _describe(self, img: Image.Image) -> str | None:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        payload = base64.b64encode(buffer.getvalue()).decode()

        try:
            resp = ollama.generate(
                model=self.model,
                prompt=self.PROMPT,
                images=[payload],
                stream=False,
                options=self.OLLAMA_OPTIONS,
            )
            return self._clean_description(resp["response"].strip())
        except Exception as e:  # pylint: disable=broad-except
            print(f"❌ Ollama error for image: {e}")
            return None

    # ------------------------------------------------------------------
    # Text post-processing helpers
    # ------------------------------------------------------------------

    def _clean_description(self, text: str) -> str:
        # Strip generic leading phrases.
        for prefix in (
            "the image shows",
            "this image shows",
            "the photo shows",
            "in this image",
            "a photo of",
            "photo of",
        ):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break

        # Replace punctuation with space, normalise whitespace.
        text = re.sub(r"[^\w\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = [w.lower() for w in text.split() if len(w) > 1]
        words = [w for w in words if w not in self.STOP_WORDS]

        # Guarantee target length.
        if len(words) < self.target_words:
            filler = ["detailed", "scene", "photo", "image", "high", "quality"]
            words.extend(filler[: self.target_words - len(words)])

        return " ".join(words[: self.target_words])

    def _to_filename(self, description: str) -> str:
        fname = description.replace(" ", "-").lower()
        fname = re.sub(r"[^a-z0-9-]", "", fname)
        fname = re.sub(r"-+", "-", fname).strip("-")
        return fname[:150] or "processed-image"

    # ------------------------------------------------------------------
    # Discovery & utility helpers
    # ------------------------------------------------------------------

    def _gather_images(self, directory: Path) -> List[Path]:
        print(f"🔍 Scanning '{directory}' for images…")
        imgs: List[Path] = []
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if name.startswith('.'):
                    continue
                p = Path(root) / name
                try:
                    if p.stat().st_size < 1024 or p.stat().st_size > 50 * 1024 * 1024:
                        continue  # Skip tiny or gigantic files (>50 MB)
                except OSError:
                    continue

                if is_image_file(p):
                    imgs.append(p)
        print(f"✅ Found {len(imgs)} images\n")
        return imgs

    def _print_summary(self, elapsed: float) -> None:
        ok = [r for r in self._results if r["success"]]
        fail = [r for r in self._results if not r["success"]]

        print("\n" + "=" * 60)
        print("🎉 All done!")
        print("=" * 60)
        print(f"Total images processed : {len(self._results)}")
        print(f"Successfully renamed   : {len(ok)}")
        print(f"Failed / skipped      : {len(fail)}")
        if elapsed:
            print(f"Elapsed time          : {elapsed:.1f}s ⇒ {len(ok)/elapsed:.1f} images/s")

        # Save JSON report next to the directory we processed (first image's parent).
        if self._results:
            out_dir = Path(self._results[0]["original"]).resolve().parent
            report = out_dir / "optimal_image_renamer_report.json"
            with report.open("w", encoding="utf-8") as fh:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "stats": {
                        "processed": len(self._results),
                        "success": len(ok),
                        "failed": len(fail),
                        "duration_seconds": elapsed,
                    },
                    "examples": ok[:10],
                }, fh, indent=2)
            print(f"📄 Detailed report written to {report}")

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------

    def _test_ollama(self, *, silent: bool = False) -> None:
        """Raise if the Ollama server is unreachable."""
        try:
            _ = ollama.generate(model=self.model, prompt="Hello", stream=False)
        except Exception as exc:  # pylint: disable=broad-except
            if not silent:
                print(f"❌ Could not connect to Ollama / model '{self.model}': {exc}")
            raise


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover – no tests in this repo
    import argparse

    parser = argparse.ArgumentParser(description="Optimal Image Renamer – in-place, AI-powered")
    parser.add_argument("directory", help="Directory that contains the images")
    parser.add_argument("--words", "-w", type=int, default=12, help="Number of words to include in the filename (default: 12)")
    parser.add_argument("--workers", type=int, help="Maximum parallel worker threads (default: #CPU cores)")
    parser.add_argument("--model", "-m", default="llava:latest", help="Ollama model to use (default: llava:latest)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts (dangerous!)")

    args = parser.parse_args()

    target = Path(args.directory).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        print(f"❌ Path '{target}' does not exist or is not a directory.")
        sys.exit(1)

    renamer = OptimalImageRenamer(model=args.model, target_words=args.words, max_workers=args.workers)
    renamer.process(target, auto_confirm=args.yes)


if __name__ == "__main__":
    main()
