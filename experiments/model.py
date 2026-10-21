"""Candidate adapter. Each CV fold gets its own prior and configuration."""
import json
from pathlib import Path
from eb_core import Predictor

_root = Path(__file__).resolve().parent
_prior = json.loads((_root / "prior.json").read_text())
_config = json.loads((_root / "config.json").read_text())
_model = Predictor(_prior, uniform_mix=_config["uniform_mix"], zero_half=_config["zero_half"])
predict = _model.predict
