"""Tests for fine-tuning module."""
import tempfile
from pathlib import Path
from finetuning.train_conspir_robust import ConspirRobustFineTuner


def test_create_experiment():
    """Test creating experiment with six-stage format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = ConspirRobustFineTuner(output_dir=tmpdir)
        config = {
            "title": "Test Experiment",
            "claim": "Fine-tuning improves conspiracy resistance",
            "unit": "model on held-out items",
            "primary_metric": "non-endorsement rate"
        }
        exp_file = tuner.create_experiment(config)
        assert exp_file.exists()
        content = exp_file.read_text()
        assert "## 1. Claim" in content
        assert "## 2. Unit and metric" in content
        assert "## 3. Controls" in content
        assert "## 4. Data" in content
        assert "## 5. Noise" in content
        assert "## 6. Readout" in content


def test_dry_run_training():
    """Test dry run training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = ConspirRobustFineTuner(output_dir=tmpdir)
        config = {"title": "Test"}
        exp_file = tuner.create_experiment(config)
        result = tuner.train(exp_file.parent, dry_run=True)
        assert result["status"] == "dry_run"
        assert result["model"] == "Qwen/Qwen2.5-3B-Instruct"
```
