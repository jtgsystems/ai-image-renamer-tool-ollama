"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_image_dir(tmp_path):
    """Create a temporary directory with sample images."""
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # Create various test images
    for i in range(5):
        img_path = image_dir / f"test_image_{i}.png"
        img = Image.new("RGB", (200, 200), color=(i * 50, 100, 150))
        img.save(img_path)

    return image_dir


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {"response": "A beautiful landscape with mountains and trees"}
