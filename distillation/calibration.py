"""Optional fixed output calibration; adds no learned model or backbone calls.

Defaults are exactly the original sampler. Calibrations must be selected on a
train-derived development split, and recorded alongside the checkpoint hash.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch.nn import functional as F


def validate_calibration(settings: Mapping[str, float] | None) -> dict[str, float]:
    defaults = dict(contrast=1.0, brightness=0.0, sharpness=0.0, saturation=1.0)
    if settings is None:
        return defaults
    unknown = set(settings) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown output calibration keys: {sorted(unknown)}")
    defaults.update({key: float(value) for key, value in settings.items()})
    if not all(math.isfinite(value) for value in defaults.values()):
        raise ValueError("Output calibration values must be finite")
    if defaults["contrast"] < 0 or defaults["saturation"] < 0:
        raise ValueError("Contrast and saturation must be nonnegative")
    return defaults


def calibrate_output(samples: torch.Tensor, settings: Mapping[str, float] | None) -> torch.Tensor:
    """Calibrate normalized RGB output before the official PNG quantization."""
    values = validate_calibration(settings)
    contrast, brightness = values["contrast"], values["brightness"]
    sharpness, saturation = values["sharpness"], values["saturation"]
    if (contrast, brightness, sharpness, saturation) == (1.0, 0.0, 0.0, 1.0):
        return samples
    x = samples
    if saturation != 1:
        weights = x.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]
        gray = (x * weights).sum(1, keepdim=True)
        x = gray + saturation * (x - gray)
    if sharpness != 0:
        blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        x = x + sharpness * (x - blur)
    return (x * contrast + brightness).clamp(-1, 1)
