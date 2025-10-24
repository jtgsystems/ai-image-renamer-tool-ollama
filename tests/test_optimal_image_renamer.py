"""Comprehensive unit tests for Optimal Image Renamer."""

from __future__ import annotations

import hashlib
import json
import subprocess

# Import from the main module
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from OPTIMALIMAGERENAMER import (
    IMAGE_EXTENSIONS,
    OptimalImageRenamer,
    is_image_file,
    md5sum,
)


class TestHelperFunctions:
    """Test helper functions."""

    def test_image_extensions_set(self):
        """Test that IMAGE_EXTENSIONS contains expected formats."""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".gif" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS

    def test_md5sum(self, tmp_path):
        """Test MD5 hash calculation."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        result = md5sum(test_file)
        expected = hashlib.md5(test_content).hexdigest()

        assert result == expected
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hash length

    def test_is_image_file_with_valid_image(self, tmp_path):
        """Test is_image_file with a valid image."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        assert is_image_file(img_path) is True

    def test_is_image_file_with_invalid_extension(self, tmp_path):
        """Test is_image_file with invalid extension."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an image")

        assert is_image_file(txt_file) is False

    def test_is_image_file_with_corrupted_image(self, tmp_path):
        """Test is_image_file with corrupted image file."""
        img_path = tmp_path / "corrupted.jpg"
        img_path.write_bytes(b"not a real image")

        assert is_image_file(img_path) is False


class TestOptimalImageRenamer:
    """Test OptimalImageRenamer class."""

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_init_with_gpu_detection(self, mock_test_ollama, mock_check_output):
        """Test initialization with GPU detection."""
        mock_check_output.return_value = "GPU 0: NVIDIA RTX 4090\nGPU 1: NVIDIA RTX 4090"
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        assert renamer.gpu_count == 2
        assert renamer.model == "llava:latest"
        assert renamer.target_words == 12

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_init_without_gpu(self, mock_test_ollama, mock_check_output):
        """Test initialization without GPU."""
        mock_check_output.side_effect = subprocess.SubprocessError()
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        assert renamer.gpu_count == 0

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_init_custom_parameters(self, mock_test_ollama, mock_check_output):
        """Test initialization with custom parameters."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer(model="llava:13b", target_words=15, max_workers=8)

        assert renamer.model == "llava:13b"
        assert renamer.target_words == 15
        assert renamer.max_workers == 8

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_clean_description(self, mock_test_ollama, mock_check_output):
        """Test description cleaning."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        # Test removing prefix
        result = renamer._clean_description("The image shows a red car on the road")
        # Check that prefix was removed (should start with content not prefix)
        assert not result.startswith("image shows")

        # Test removing stop words with longer description
        result = renamer._clean_description(
            "A beautiful sunset over the ocean with amazing colors and reflections"
        )
        words = result.split()
        assert "the" not in words
        assert len(words) <= renamer.target_words

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_to_filename(self, mock_test_ollama, mock_check_output):
        """Test filename conversion."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        result = renamer._to_filename("beautiful sunset over ocean")
        assert result == "beautiful-sunset-over-ocean"
        assert " " not in result
        assert result.islower()

        # Test with special characters
        result = renamer._to_filename("test@#$% image!")
        assert "@" not in result
        assert "#" not in result

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_load_image_rgb(self, mock_test_ollama, mock_check_output, tmp_path):
        """Test loading RGB image."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        loaded = renamer._load_image(img_path)
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_load_image_rgba(self, mock_test_ollama, mock_check_output, tmp_path):
        """Test loading RGBA image (converts to RGB)."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        img_path = tmp_path / "test.png"
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img.save(img_path)

        loaded = renamer._load_image(img_path)
        assert loaded.mode == "RGB"

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_load_image_resize(self, mock_test_ollama, mock_check_output, tmp_path):
        """Test that large images are resized."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        img_path = tmp_path / "large.png"
        img = Image.new("RGB", (5000, 5000), color="blue")
        img.save(img_path)

        loaded = renamer._load_image(img_path)
        assert max(loaded.size) <= 4096

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_gather_images(self, mock_test_ollama, mock_check_output, tmp_path):
        """Test gathering images from directory."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        # Create test images large enough to pass size filter (>1KB)
        for i in range(3):
            img_path = tmp_path / f"test{i}.png"
            # Create larger images to ensure file size > 1KB
            img = Image.new("RGB", (500, 500), color=(i * 50, 100, 150))
            img.save(img_path, "PNG", compress_level=0)  # No compression for larger file

        # Create a text file (should be ignored)
        (tmp_path / "text.txt").write_text("not an image")

        # Create a hidden file (should be ignored)
        hidden_img = tmp_path / ".hidden.png"
        Image.new("RGB", (500, 500)).save(hidden_img, "PNG", compress_level=0)

        images = renamer._gather_images(tmp_path)
        assert len(images) == 3

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_process_one_duplicate_detection(self, mock_test_ollama, mock_check_output, tmp_path):
        """Test duplicate detection in process_one."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        # Create an image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        # Add hash to seen set
        file_hash = md5sum(img_path)
        renamer._hashes_seen.add(file_hash)

        # Try to process
        result = renamer._process_one(img_path)

        assert result["success"] is False
        assert result["error"] == "duplicate-image"

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_cleanup_server(self, mock_test_ollama, mock_check_output):
        """Test server cleanup."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()

        # Mock a process
        mock_proc = MagicMock()
        renamer._ollama_proc = mock_proc

        renamer._cleanup_server()

        mock_proc.terminate.assert_called_once()

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    def test_cleanup_server_no_process(self, mock_test_ollama, mock_check_output):
        """Test cleanup when no server process exists."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None

        renamer = OptimalImageRenamer()
        renamer._ollama_proc = None

        # Should not raise an exception
        renamer._cleanup_server()


class TestIntegration:
    """Integration tests."""

    @patch("OPTIMALIMAGERENAMER.subprocess.check_output")
    @patch("OPTIMALIMAGERENAMER.OptimalImageRenamer._test_ollama")
    @patch("OPTIMALIMAGERENAMER.ollama.generate")
    def test_full_process_with_auto_confirm(
        self, mock_generate, mock_test_ollama, mock_check_output, tmp_path
    ):
        """Test full processing workflow."""
        mock_check_output.return_value = ""
        mock_test_ollama.return_value = None
        mock_generate.return_value = {"response": "A beautiful red car on the road"}

        renamer = OptimalImageRenamer()

        # Create test images large enough to pass size filter (>1KB)
        for i in range(3):
            img_path = tmp_path / f"IMG_{i:04d}.png"
            img = Image.new("RGB", (500, 500), color="red")
            img.save(img_path, "PNG", compress_level=0)

        # Process with auto-confirm
        renamer.process(tmp_path, auto_confirm=True)

        # Check that files were renamed
        remaining_files = list(tmp_path.glob("*.png"))
        assert len(remaining_files) == 3

        # Check that report was created
        report_path = tmp_path / "optimal_image_renamer_report.json"
        assert report_path.exists()

        # Validate report content
        with report_path.open() as f:
            report = json.load(f)
        assert "stats" in report
        assert report["stats"]["processed"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
