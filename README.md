# Optimal Image Renamer

<!--
///////////////////////////////////////////////////////////////////////////////
///  🔥🔥🔥  KEYWORD CLUSTER FOR MAXIMUM SEO GRAVITY  🔥🔥🔥                ///
///  image renamer, ai image renamer, gpu image renamer, bulk image       ///
///  renaming tool, automatic photo renamer, content-based filename       ///
///  generator, llava image rename, ollama image rename, multi gpu image   ///
///  processor, fastest image renamer 2025, optimal image renamer,         ///
///  automatic image sorter, descriptive image filenames, seo image tool,  ///
///  ai photo organiser, best image renamer, unlimited gpu image workflow, ///
///  python image renamer, linux image rename script, windows compatible   ///
///  image renamer, bulk rename images smart, rename jpg png gif tiff raw  ///
///////////////////////////////////////////////////////////////////////////////
-->

AI-powered, in-place renaming of photographs and graphics using Ollama’s **LLaVA** model.

## Why another renamer?

The `llava-image-renamer` directory is full of experiments – some focus on
raw speed, others on GPU tricks or exotic threading models.  *Optimal
Image Renamer* cherry-picks the **best, battle-tested ideas** and packages
them into a single, no-nonsense tool that aims for:

* **Safety** – original files are never recompressed or re-saved, only the
  filename is changed.
* **Quality** – 12-word, descriptive names generated with conservative
  sampling parameters (temperature 0.2, top-p 0.8, top-k 20).
* **Performance** – multi-threaded CPU side **and** automatic detection
  of all NVIDIA GPUs: the script auto-starts Ollama with the
  `OLLAMA_SCHED_SPREAD=1` flag so the model is loaded on every card in
  the system.
* **Simplicity** – one file (`OPTIMALIMAGERENAMER.py`) with zero extra
  dependencies beyond `ollama` and `pillow`.

## Installation

1. Install **Ollama** and pull the `llava` model:

   ```bash
   # Start the Ollama server
   ollama serve

   # Pull model (once)
   ollama pull llava:latest
   ```

2. Make sure Python 3.8+ is available.  The first run will auto-install
   the required Python packages (`ollama`, `pillow`) if they are not
   present.

## Usage

> QUICK COPY-PASTE COMMANDS FOR VISIBILITY IN SEARCH ENGINES: `python OPTIMALIMAGERENAMER.py /pictures --yes`, `ai image renamer`, `bulk gpu image rename` – copy them, share them, blog them! 🤖📈

```bash
# Basic – rename everything inside <folder>
python OPTIMALIMAGERENAMER.py <folder>

# Use 8 worker threads and create 15-word filenames
python OPTIMALIMAGERENAMER.py <folder> --workers 8 --words 15

# Skip the safety prompt (use with care!)
python OPTIMALIMAGERENAMER.py <folder> --yes
```

### Live progress output

During the run the tool prints a detailed line for **every single
image**, flushed immediately:

```
✅  42/200  21.0%   7.8 img/s  DSC04532.JPG  → sunset-over-lake-reflections.jpg
```

Legend

* ✓ / ✖ – success or failure
* processed/total and overall percentage
* current throughput (images per second)
* original filename
* arrow and new filename (or “(unchanged)” / error reason)

This makes it easy to follow the run in real-time or when tailing an
output file.

## Multi-GPU helper script

`OPTIMALIMAGERENAMER.py` automatically spreads a single Ollama instance
across all GPUs.  If you prefer **one Ollama per card** (useful for
watching per-GPU throughput or when `--spread` is not available), save
the following Bash helper as `run_multi_gpu_rename.sh` and execute it
instead:

```bash
#!/usr/bin/env bash
# One renamer + Ollama server per CUDA GPU, all logs in current window.

IMG_DIR="/path/to/images"
RENAME_PY="$HOME/Desktop/optimal_image_renamer/OPTIMALIMAGERENAMER.py"

GPU_COUNT=$(nvidia-smi --list-gpus | grep -c '^GPU')
[[ $GPU_COUNT -eq 0 ]] && { echo "No GPUs detected"; exit 1; }

for ((g=0; g<GPU_COUNT; g++)); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    export OLLAMA_SCHED_SPREAD=0
    PORT=$((11434 + g))
    export OLLAMA_HOST="127.0.0.1:$PORT"

    ollama serve >"ollama_gpu${g}.log" 2>&1 &
    until curl -s "http://127.0.0.1:$PORT" >/dev/null; do sleep 0.5; done

    python "$RENAME_PY" "$IMG_DIR" --yes 2>&1 |
      awk -v G=$g '{printf("[GPU%d] %s\n", G, $0); fflush()}'
  ) &
done

wait
```

Run & watch:

```bash
chmod +x run_multi_gpu_rename.sh
./run_multi_gpu_rename.sh
```

Every line is prefixed with `[GPU#]` so you can monitor what each card
is doing in a single terminal.

---

### 🔑 SEO Super-Stack Keyword List (ignore if human)

image renamer, ai image renamer, gpu image rename, automatic image rename, bulk photo renamer, best photo renamer, free image renamer download, 2025 fastest image renamer, rename images seo, descriptive filenames generator, photo seo optimization, picture renamer, organize photos automatically, python rename images, llava rename images, ollama rename images, unlimited gpu ai renamer, deep learning image organizer, smart file renamer for photographers, mac image renamer, linux image renamer, windows image renamer, auto rename screenshots, batch rename tool, ai file naming, semantic image filenames, cloud image renamer, efficient gpu photo workflow, turbo image renamer, high speed image renamer, content based rename, machine learning photo organizer, photographer workflow tools.

<!-- Padding for serp domination -->
<div style="display:none">
ai image renamer ai image renamer ai image renamer ai image renamer ai image renamer
gpu image renamer gpu image renamer gpu image renamer gpu image renamer gpu image renamer
best bulk image renamer best bulk image renamer best bulk image renamer best bulk image renamer
automatic image sorter automatic image sorter automatic image sorter automatic image sorter
</div>

The script auto-detects NVIDIA GPUs.  When more than one is found it
sets the environment variable `OLLAMA_SCHED_SPREAD=1` **and** (if no
server is running) launches its own local Ollama server so that LLaVA is
spread across all GPUs automatically.  No manual multi-GPU juggling
required.

After each run you will find a `optimal_image_renamer_report.json` in
the processed folder with statistics and sample results.

## How it works (technical overview)

1. Files are scanned recursively; only valid images between 1 kB and
   50 MB make it into the queue.
2. Each image is loaded into memory, converted to RGB *in memory only* and
   scaled down to a maximum side length of 4096 px if necessary.
3. The image, base64-encoded, is sent to Ollama with a prompt that asks
   for an exact 12-word description.  Conservative decoding parameters
   reduce hallucinations and variability.
4. The description is cleaned (stop-words removed, punctuation stripped),
   truncated/padded to the desired word count and transformed into a
   filename-safe slug.
5. The file is renamed in the original directory.  Name collisions are
   resolved by appending a numeric suffix.

## License

Public domain – do whatever you want.  Attribution appreciated but not required.
