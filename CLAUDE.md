# Claude Code - AI Image Renamer Tool

This file contains project architecture, workflow documentation, and development guidelines for the AI-powered image renaming tool.

---

## Project Overview

**AI Image Renamer Tool** is a high-performance, GPU-accelerated Python application that intelligently renames image files using Ollama's qwen3-vl vision-language model. The tool generates descriptive, SEO-friendly filenames by analyzing image content with AI.

### Key Value Propositions

- **100% Safe**: Only renames files, never modifies image data
- **AI-Powered**: Uses qwen3-vl vision model for accurate content descriptions
- **Multi-GPU Ready**: Automatic NVIDIA GPU detection with workload distribution
- **Zero Manual Configuration**: Auto-installs dependencies on first run
- **Production-Grade**: Includes duplicate detection, collision resolution, and comprehensive error handling

### Primary Use Cases

1. **Photographers & Content Creators**: Organize thousands of photos with descriptive names
2. **E-commerce**: SEO-optimize product image catalogs
3. **Web Developers**: Semantic naming for better accessibility and search rankings
4. **Digital Asset Management**: Enterprise-scale photo library organization

---

## Repository Information

- **GitHub**: `jtgsystems/ai-image-renamer-tool-ollama`
- **License**: Public Domain
- **Primary Language**: Python 3.8+
- **File Structure**: Single-file architecture for portability

---

## Directory Structure

```
ai-image-renamer-tool-ollama/
├── OPTIMALIMAGERENAMER.py    # Main application (single-file architecture)
├── README.md                  # User documentation with examples
├── requirements.txt           # Python dependencies (ollama, pillow)
├── banner.png                 # Repository banner image
└── .git/                      # Git repository metadata
```

### Key Files

#### OPTIMALIMAGERENAMER.py
- **Lines of Code**: 460
- **Architecture**: Object-oriented with `OptimalImageRenamer` facade class
- **Dependencies**: `ollama>=0.6.1`, `Pillow>=12.0.0`, standard library (pathlib, concurrent.futures, hashlib, base64)
- **Entry Point**: CLI via `argparse` in `main()` function

#### requirements.txt
```
ollama
pillow
```

#### README.md
- **Purpose**: User-facing documentation with installation, usage, benchmarks, and troubleshooting
- **SEO Keywords**: ai image renamer, gpu image processing, bulk photo organizer, seo image optimization
- **Includes**: Quick start guide, multi-GPU setup scripts, performance benchmarks

---

## Technology Stack

### Core Technologies

#### Python 3.8+
- **Standard Library Modules**:
  - `pathlib`: Cross-platform file path handling
  - `concurrent.futures.ThreadPoolExecutor`: Multithreading for parallel processing
  - `hashlib`: MD5 duplicate detection
  - `base64`: Image encoding for API transmission
  - `argparse`: CLI interface
  - `json`: Report generation
  - `subprocess`: Ollama server management and GPU detection

#### AI/ML Framework
- **Ollama**: Local LLM inference server
  - Model: `qwen3-vl` (default, configurable)
  - API: Python client library via `ollama` package
  - Vision-Language Model: Processes images and generates text descriptions

#### Image Processing
- **Pillow (PIL)**: Python Imaging Library
  - Image validation (`Image.verify()`)
  - Format conversion (RGBA → RGB, color space normalization)
  - Thumbnail generation (max 4096px to avoid memory issues)
  - JPEG encoding for API transmission (quality=95)

### GPU Acceleration

#### NVIDIA CUDA
- **Detection**: `nvidia-smi --list-gpus`
- **Multi-GPU Strategy**: `OLLAMA_SCHED_SPREAD=1` environment variable
- **Automatic Scaling**: Workload distributed across all detected GPUs
- **Fallback**: CPU inference if no GPUs detected

### Performance Optimizations

#### Multithreading
- **Default Workers**: `max(os.cpu_count() or 4, 4)`
- **Configurable**: `--workers N` CLI flag
- **Thread Pool**: `concurrent.futures.ThreadPoolExecutor`

#### Memory Management
- **In-Memory Processing**: Images never recompressed/saved during analysis
- **Size Limits**: Skip files <1KB or >50MB
- **Thumbnail Scaling**: Max 4096px to prevent memory overflow
- **Hash-based Deduplication**: MD5 checksums prevent duplicate processing

---

## Development Workflow

### Installation

#### Prerequisites
1. **Ollama Server**:
   ```bash
   # Install Ollama (see ollama.ai)
   ollama serve
   ollama pull qwen3-vl
   ```

2. **Python 3.8+**:
   ```bash
   python3 --version  # Verify 3.8+
   ```

#### First Run (Auto-Install)
```bash
git clone git@github.com:jtgsystems/ai-image-renamer-tool-ollama.git
cd ai-image-renamer-tool-ollama
python3 OPTIMALIMAGERENAMER.py /path/to/images
```

**Auto-Install Behavior**: Script detects missing dependencies and runs:
```python
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ollama", "pillow"])
```

### CLI Commands

#### Basic Usage
```bash
# Rename all images in a directory (with confirmation prompt)
python3 OPTIMALIMAGERENAMER.py /path/to/images

# Skip safety confirmation (YOLO mode)
python3 OPTIMALIMAGERENAMER.py /path/to/images --yes

# Custom word count (default: 12 words)
python3 OPTIMALIMAGERENAMER.py /path/to/images --words 15

# Custom worker threads (default: CPU core count)
python3 OPTIMALIMAGERENAMER.py /path/to/images --workers 8

# Use different Ollama model
python3 OPTIMALIMAGERENAMER.py /path/to/images --model qwen3-vl:13b
```

#### Advanced Multi-GPU Setup
```bash
# Create launcher script (see README for full script)
chmod +x run_multi_gpu_rename.sh
./run_multi_gpu_rename.sh
```

### Configuration

#### AI Model Parameters (Hardcoded in Class)
```python
OLLAMA_OPTIONS = {
    "temperature": 0.2,        # Low temperature = less hallucination
    "top_p": 0.8,              # Nucleus sampling threshold
    "top_k": 20,               # Top-K sampling (focused vocabulary)
    "repeat_penalty": 1.05,    # Penalize repetitive tokens
    "num_predict": 60,         # Max tokens to generate
}
```

**Rationale**: Conservative parameters chosen after benchmarking to balance quality and speed.

#### Prompt Engineering
```python
PROMPT = (
    "Describe this image in **exactly 12 words**. "
    "Focus on the main subject, important objects and the setting. "
    "Be specific, use descriptive adjectives, avoid redundancy."
)
```

#### Stop Words (Filtered from Filenames)
```python
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "this", "that", "these", "those", "there", "where",
    "when", "how", "what", "which",
}
```

---

## Application Architecture

### Class Structure

#### OptimalImageRenamer (Main Facade)
```
OptimalImageRenamer
├── __init__(model, target_words, max_workers)
├── process(directory, auto_confirm)          # Public API entry point
├── _process_one(path)                        # Single image pipeline
├── _load_image(path)                         # Image preprocessing
├── _describe(img)                            # AI inference via Ollama
├── _clean_description(text)                  # Text normalization
├── _to_filename(description)                 # Slug generation
├── _gather_images(directory)                 # Recursive file discovery
├── _print_summary(elapsed)                   # Stats and JSON report
├── _detect_gpu_count()                       # NVIDIA GPU detection
├── _ensure_ollama_server_ready()             # Auto-start Ollama server
├── _test_ollama(silent)                      # Connectivity check
└── _cleanup_server()                         # Cleanup on exit (atexit hook)
```

### Processing Pipeline

#### Phase 1: Discovery
1. Recursive scan of target directory (`os.walk`)
2. Filter by extension (`IMAGE_EXTENSIONS` set)
3. Size validation (1KB - 50MB range)
4. Pillow verification (`Image.verify()`)
5. Skip hidden files (starting with `.`)

#### Phase 2: Preprocessing
1. Open image with Pillow (`Image.open`)
2. Color space normalization:
   - RGBA → RGB with white background
   - All other modes → RGB
3. Thumbnail scaling (max 4096px, LANCZOS resampling)
4. In-memory copy to prevent file handle leaks

#### Phase 3: AI Inference
1. JPEG encode at quality=95 → BytesIO buffer
2. Base64 encode for API transmission
3. Ollama API call with:
   - Model: `qwen3-vl`
   - Prompt: 12-word description request
   - Images: [base64_payload]
   - Options: Conservative temperature/sampling
4. Extract `response` field from JSON response

#### Phase 4: Text Processing
1. Strip generic prefixes ("the image shows", "photo of", etc.)
2. Remove punctuation (replace with spaces)
3. Normalize whitespace
4. Lowercase all words
5. Filter stop words
6. Pad with filler words if under target count
7. Truncate to target word count (12 by default)

#### Phase 5: Filename Generation
1. Replace spaces with hyphens
2. Remove non-alphanumeric characters (keep hyphens)
3. Collapse multiple hyphens
4. Strip leading/trailing hyphens
5. Truncate to 150 characters max
6. Append original extension (lowercased)

#### Phase 6: Collision Resolution
1. Check if target filename exists
2. If exists and != original path:
   - Append `-01`, `-02`, etc.
   - Increment until unique filename found
3. Perform atomic `os.rename()`

#### Phase 7: Deduplication
- MD5 hash computed for each image
- Hash stored in `_hashes_seen` set
- Skip processing if hash already seen (marks as "duplicate-image" error)

### Multithreading Model

```python
with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    futures = {executor.submit(self._process_one, img): img for img in images}
    for idx, future in enumerate(as_completed(futures), 1):
        res = future.result()
        # Real-time progress reporting
        # Status: ✅/❌  Progress: 42/200 21.0%  Speed: 7.8 img/s
```

**Key Features**:
- Thread-safe operations (no shared mutable state except `_hashes_seen` set)
- Real-time progress updates as futures complete
- Exception handling per-image (failures don't crash entire batch)

---

## Output & Reporting

### Console Output (Real-Time)

```
🔍 Scanning '/path/to/images' for images…
✅ Found 200 images

🚀 Starting rename of 200 images using up to 8 threads…

✅   42/200  21.0%   7.8 img/s  DSC04532.JPG  → sunset-over-lake-reflections.jpg
✅   43/200  21.5%   7.9 img/s  IMG_1234.PNG  → golden-retriever-playing-park.png
❌   44/200  22.0%   8.1 img/s  corrupt.jpg   ✖ cannot identify image file

============================================================
🎉 All done!
============================================================
Total images processed : 200
Successfully renamed   : 198
Failed / skipped      : 2
Elapsed time          : 25.6s ⇒ 7.7 images/s
📄 Detailed report written to /path/to/images/optimal_image_renamer_report.json
```

### JSON Report (`optimal_image_renamer_report.json`)

```json
{
  "timestamp": "2025-10-26T03:45:12Z",
  "stats": {
    "processed": 200,
    "success": 198,
    "failed": 2,
    "duration_seconds": 25.6
  },
  "examples": [
    {
      "original": "/path/to/images/DSC04532.JPG",
      "success": true,
      "new_name": "/path/to/images/sunset-over-lake-reflections.jpg",
      "description": "sunset mountain lake golden hour reflection",
      "error": null
    }
  ]
}
```

**Location**: Saved in same directory as processed images.

---

## Performance Benchmarks

### Single GPU Performance
- **Speed**: 8-12 images/second (NVIDIA RTX series)
- **Memory**: <2GB VRAM per worker thread
- **Accuracy**: 95%+ relevant, descriptive filenames

### Multi-GPU Scaling
- **2x GPUs**: ~16-20 images/second
- **4x GPUs**: ~32-40 images/second
- **Scaling**: Near-linear with `OLLAMA_SCHED_SPREAD=1`

### CPU Fallback
- **Speed**: 0.5-1 images/second (much slower)
- **Use Case**: Development/testing on non-GPU systems

---

## Error Handling & Edge Cases

### Handled Scenarios

#### Missing Dependencies
- **Detection**: `ModuleNotFoundError` catch block
- **Auto-Fix**: `pip install ollama pillow`
- **User Experience**: Transparent auto-installation

#### Ollama Server Not Running
- **Detection**: Connection test fails
- **Auto-Fix**: `subprocess.Popen(['ollama', 'serve'])`
- **Retry Logic**: 20 attempts with 0.5s delay
- **Fallback**: Exit with clear error message

#### Invalid Image Files
- **Detection**: `Image.verify()` or `Image.open()` exception
- **Handling**: Skip file, mark as failed in report
- **No Crash**: Isolated to single thread, doesn't affect batch

#### Filename Collisions
- **Detection**: `new_path.exists()`
- **Resolution**: Append `-01`, `-02`, etc.
- **Safety**: Never overwrites existing files

#### Duplicate Images
- **Detection**: MD5 hash in `_hashes_seen` set
- **Handling**: Skip processing, mark as "duplicate-image"
- **Performance**: Saves redundant AI calls

#### Large/Small Files
- **Filter**: Skip <1KB (metadata/icons) or >50MB (gigantic RAW files)
- **Rationale**: Avoid processing non-photos and memory issues

#### RGBA/Non-RGB Images
- **Conversion**: RGBA → RGB with white background
- **Normalization**: All modes converted to RGB
- **Compatibility**: Ensures consistent AI inference

### Error Messages

```python
# Per-image errors stored in results dictionary
{
  "error": "duplicate-image"      # MD5 hash collision
  "error": "no-description"       # AI returned empty response
  "error": "cannot identify..."   # Pillow can't open file
  "error": "<exception message>"  # General exception
}
```

---

## Testing Approach

### Current State
- **No automated tests**: Single-file architecture prioritizes portability
- **Manual testing**: Validated against real-world image collections

### Recommended Testing Strategy

#### Unit Tests (If Expanding)
```python
# Test filename sanitization
assert _to_filename("Sunset! Over Lake?") == "sunset-over-lake"

# Test stop word filtering
assert _clean_description("The cat on the mat") == "cat mat ..."

# Test collision resolution
assert resolves_filename_conflict("image.jpg", "image-01.jpg")
```

#### Integration Tests
```python
# Test Ollama connectivity
def test_ollama_connection():
    renamer = OptimalImageRenamer()
    renamer._test_ollama()  # Should not raise

# Test GPU detection
def test_gpu_detection():
    count = OptimalImageRenamer._detect_gpu_count()
    assert count >= 0
```

#### End-to-End Tests
```bash
# Create test image directory
mkdir test_images
cp sample1.jpg sample2.png test_images/

# Run renamer
python OPTIMALIMAGERENAMER.py test_images --yes

# Verify outputs
ls test_images/  # Should show descriptive filenames
cat test_images/optimal_image_renamer_report.json
```

---

## Known Issues & Troubleshooting

### Issue: "No GPUs detected" (But GPUs Exist)

**Cause**: NVIDIA drivers or CUDA not installed/configured

**Fix**:
```bash
# Verify nvidia-smi works
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall NVIDIA drivers if needed
```

### Issue: Ollama Connection Errors

**Cause**: Ollama server not running or wrong port

**Fix**:
```bash
# Manually start Ollama
ollama serve

# Verify model is downloaded
ollama list | grep qwen3-vl

# Pull model if missing
ollama pull qwen3-vl
```

### Issue: Out of Memory Errors

**Cause**: Too many worker threads for available GPU VRAM

**Fix**:
```bash
# Reduce worker count
python OPTIMALIMAGERENAMER.py /path --workers 4

# Close other GPU applications
nvidia-smi  # Check what's using VRAM
```

### Issue: Slow Processing Speed

**Cause**: Model not downloaded, CPU inference, or network issues

**Fix**:
```bash
# Ensure model is local (not streaming from network)
ollama pull qwen3-vl

# Check GPU is being used
nvidia-smi  # Should show ollama process

# Increase workers if CPU/GPU underutilized
python OPTIMALIMAGERENAMER.py /path --workers 16
```

### Issue: Generic/Poor Filenames

**Cause**: Low-quality images, abstract content, or AI limitations

**Fix**:
```bash
# Try larger model for better descriptions
python OPTIMALIMAGERENAMER.py /path --model qwen3-vl:13b

# Increase word count for more specificity
python OPTIMALIMAGERENAMER.py /path --words 20
```

---

## Development Guidelines

### Code Style

- **Language**: Python 3.8+ with type hints where beneficial
- **Formatting**: 4-space indentation, max line length ~100 chars
- **Docstrings**: Module-level and class-level docstrings
- **Comments**: Inline comments for complex logic

### Adding Features

#### New CLI Flags
```python
# In main() function
parser.add_argument("--new-flag", help="Description")
```

#### New AI Models
```python
# Modify OLLAMA_OPTIONS or PROMPT
# Test with: --model <new_model_name>
```

#### Custom Filename Formats
```python
# Modify _to_filename() method
# Example: Add underscores instead of hyphens
fname = description.replace(" ", "_").lower()
```

### Git Workflow

```bash
# Clone repository
git clone git@github.com:jtgsystems/ai-image-renamer-tool-ollama.git

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, commit
git add .
git commit -m "Add feature: description"

# Push to GitHub
git push origin feature/your-feature-name

# Open pull request on GitHub
```

---

## Performance Optimization Tips

### GPU Utilization

1. **Multi-GPU Setup**: Use `run_multi_gpu_rename.sh` for maximum throughput
2. **Worker Tuning**: Set `--workers` to 2-4x GPU count for optimal queue depth
3. **Batch Processing**: Process large directories (1000+ images) to amortize startup costs

### Memory Optimization

1. **Image Scaling**: Already optimized at 4096px max
2. **Thread Count**: Don't exceed `2 × CPU_CORES` on shared systems
3. **File Filtering**: Adjust size limits in `_gather_images()` if needed

### AI Quality vs Speed

```python
# Faster (less accurate)
OLLAMA_OPTIONS = {"temperature": 0.5, "top_k": 10}

# Slower (more accurate)
OLLAMA_OPTIONS = {"temperature": 0.1, "top_k": 40}
```

---

## Roadmap & Next Steps

### Potential Enhancements

#### Short-Term (Low Effort, High Impact)
- [ ] Add `--dry-run` flag to preview renames without applying
- [ ] Support custom prompt templates via `--prompt` flag
- [ ] Add progress bar using `tqdm` library
- [ ] CSV export of rename mappings (old → new)

#### Medium-Term (Moderate Effort)
- [ ] Web UI for drag-and-drop image renaming
- [ ] Support for video file renaming (use first frame)
- [ ] Batch mode: Process multiple directories sequentially
- [ ] Undo functionality (save original filenames to JSON)

#### Long-Term (High Effort)
- [ ] Integration with cloud storage (S3, Google Drive, Dropbox)
- [ ] REST API for remote image renaming
- [ ] Docker container for zero-install deployment
- [ ] Support for other vision models (CLIP, BLIP, etc.)

### Community Contributions

**Welcome contributions**:
- Performance benchmarks on different hardware
- Bug fixes and error handling improvements
- Documentation enhancements
- New use case examples

---

## Dependencies Reference

### Python Packages

#### ollama (Required)
- **Purpose**: Python client for Ollama API
- **Install**: `pip install ollama`
- **Docs**: https://github.com/ollama/ollama-python

#### pillow (Required)
- **Purpose**: Image file validation and preprocessing
- **Install**: `pip install pillow`
- **Docs**: https://pillow.readthedocs.io/

### External Tools

#### Ollama (Required)
- **Purpose**: Local LLM inference server
- **Install**: https://ollama.ai/download
- **Models**: `ollama pull qwen3-vl`

#### NVIDIA GPU Drivers (Optional, Recommended)
- **Purpose**: GPU acceleration for 10x+ speed improvement
- **Install**: https://www.nvidia.com/Download/index.aspx
- **Verify**: `nvidia-smi`

---

## Support & Resources

### Documentation
- **README.md**: User-facing installation and usage guide
- **CLAUDE.md**: This file (developer/AI assistant reference)

### Community
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share use cases

### Related Projects
- **Ollama**: https://ollama.ai
- **qwen3-vl Model**: https://qwen3-vl-vl.github.io
- **Pillow**: https://pillow.readthedocs.io

---

## Quick Reference Commands

```bash
# Basic usage
python3 OPTIMALIMAGERENAMER.py /path/to/images

# Production use (skip confirmation)
python3 OPTIMALIMAGERENAMER.py /path/to/images --yes

# High-quality mode (more words, larger model)
python3 OPTIMALIMAGERENAMER.py /path/to/images --words 20 --model qwen3-vl:13b

# Maximum performance (multi-GPU)
./run_multi_gpu_rename.sh  # See README for script

# Check GPU availability
nvidia-smi

# Verify Ollama is running
curl http://localhost:11434

# Pull/update qwen3-vl model
ollama pull qwen3-vl
```

---

## Notes for AI Assistants (Claude Code)

### When Working With This Project

1. **Read Files First**: Always read `OPTIMALIMAGERENAMER.py` before making changes
2. **Test Carefully**: Changes to AI parameters affect all rename operations
3. **Preserve Safety**: Don't remove duplicate detection or collision resolution
4. **Document Changes**: Update this file when modifying architecture

### Common Tasks

**Add New CLI Flag**:
1. Modify `argparse` setup in `main()`
2. Pass argument to `OptimalImageRenamer.__init__()` or `process()`
3. Update README.md with flag documentation

**Change AI Model**:
1. Update `DEFAULT_MODEL` or accept via `--model` flag
2. Test with sample images
3. Document performance differences

**Modify Filename Format**:
1. Edit `_to_filename()` method
2. Consider backward compatibility
3. Add examples to README.md

### Development Environment Setup

```bash
# Clone repository
cd ~/Desktop
git clone git@github.com:jtgsystems/ai-image-renamer-tool-ollama.git
cd ai-image-renamer-tool-ollama

# Ensure Ollama is running
ollama serve &
ollama pull qwen3-vl

# Test with sample images
mkdir test_images
# Add test images
python3 OPTIMALIMAGERENAMER.py test_images --yes
```

---

*Last Updated: 2025-12-26*
*Project: AI Image Renamer Tool (Ollama + qwen3-vl)*
*Maintainer: JTGSYSTEMS*

## Project Dependencies

*Last updated: 2025-10-26*

This is a Python project. See the following files for dependency information:

- `requirements.txt` - Python package dependencies

