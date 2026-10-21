"""Run the original synthetic experiment checks as part of release tests."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from test_experiments import InferenceTests, TrainingTests, PipelineTests
