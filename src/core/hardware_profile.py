"""
Hardware Profile Module — Auto-detects GPU, VRAM, RAM and CPU cores.
Classifies system into ULTRA_LOW / LOW / MID / HIGH tiers and returns
a HardwareProfile dataclass with all runtime settings.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


class HardwareTier(str, Enum):
    """Hardware capability tier."""
    ULTRA_LOW = "ULTRA_LOW"
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


@dataclass
class HardwareProfile:
    """Runtime hardware settings derived from auto-detection."""
    tier: HardwareTier
    device: str                  # 'cuda' or 'cpu'
    gpu_name: str                # e.g. 'NVIDIA GeForce RTX 2050'
    vram_gb: float               # GPU VRAM in GB (0.0 if CPU-only)
    ram_gb: float                # System RAM in GB
    cpu_cores: int               # Logical CPU core count
    yolo_model: str              # 'yolov8n' or 'yolov8s'
    yolo_img_size: int           # 416 or 640
    half_precision: bool         # FP16 inference flag
    capture_interval_ms: int     # ms between screen captures
    num_workers: int             # parallel CPU workers for indicators


def _detect_gpu() -> tuple[bool, str, float]:
    """
    Detect CUDA GPU availability, name, and VRAM.

    Returns:
        (has_cuda, gpu_name, vram_gb)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            vram_gb = round(vram_bytes / (1024 ** 3), 2)
            return True, gpu_name, vram_gb
    except ImportError:
        logger.warning("PyTorch not installed — falling back to CPU")
    except Exception as e:
        logger.warning("GPU detection failed: %s", e)

    return False, "No GPU", 0.0


def _classify_tier(has_cuda: bool, vram_gb: float, ram_gb: float) -> HardwareTier:
    """
    Classify hardware into a tier based on GPU and RAM.

    Rules (from PRD):
        ULTRA_LOW: no CUDA or VRAM < 1.5 GB
        LOW:       CUDA, 1.5–3 GB VRAM, 8 GB RAM
        MID:       CUDA, 3–6 GB VRAM, 16 GB RAM
        HIGH:      CUDA, 6+ GB VRAM
    """
    if not has_cuda or vram_gb < 1.5:
        return HardwareTier.ULTRA_LOW

    if vram_gb < 3.0:
        return HardwareTier.LOW

    if vram_gb < 6.0:
        return HardwareTier.MID

    return HardwareTier.HIGH


def _tier_settings(tier: HardwareTier) -> dict:
    """Return model/inference settings per tier."""
    settings = {
        HardwareTier.ULTRA_LOW: {
            "yolo_model": "yolov8n",
            "yolo_img_size": 416,
            "half_precision": False,
            "capture_interval_ms": 1000,
            "num_workers": 2,
        },
        HardwareTier.LOW: {
            "yolo_model": "yolov8n",
            "yolo_img_size": 416,
            "half_precision": True,
            "capture_interval_ms": 750,
            "num_workers": 2,
        },
        HardwareTier.MID: {
            "yolo_model": "yolov8s",
            "yolo_img_size": 640,
            "half_precision": True,
            "capture_interval_ms": 500,
            "num_workers": 4,
        },
        HardwareTier.HIGH: {
            "yolo_model": "yolov8s",
            "yolo_img_size": 640,
            "half_precision": True,
            "capture_interval_ms": 500,
            "num_workers": 4,
        },
    }
    return settings[tier]


def detect_hardware(force_tier: Optional[str] = None) -> HardwareProfile:
    """
    Auto-detect hardware and return a fully populated HardwareProfile.

    Args:
        force_tier: Override auto-detection with a specific tier string
                    (e.g. 'LOW', 'MID'). Used for testing or config override.

    Returns:
        HardwareProfile dataclass with all runtime settings.
    """
    has_cuda, gpu_name, vram_gb = _detect_gpu()
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    cpu_cores = psutil.cpu_count(logical=True) or 4

    # Allow manual tier override
    if force_tier and force_tier in HardwareTier.__members__:
        tier = HardwareTier(force_tier)
        logger.info("Tier forced to %s (override)", tier.value)
    else:
        tier = _classify_tier(has_cuda, vram_gb, ram_gb)

    device = "cuda" if has_cuda and tier != HardwareTier.ULTRA_LOW else "cpu"
    settings = _tier_settings(tier)

    profile = HardwareProfile(
        tier=tier,
        device=device,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        yolo_model=settings["yolo_model"],
        yolo_img_size=settings["yolo_img_size"],
        half_precision=settings["half_precision"],
        capture_interval_ms=settings["capture_interval_ms"],
        num_workers=settings["num_workers"],
    )

    logger.info(
        "Hardware Profile → Tier: %s | Device: %s | GPU: %s | "
        "VRAM: %.1f GB | RAM: %.1f GB | Model: %s | ImgSize: %d | FP16: %s",
        profile.tier.value,
        profile.device,
        profile.gpu_name,
        profile.vram_gb,
        profile.ram_gb,
        profile.yolo_model,
        profile.yolo_img_size,
        profile.half_precision,
    )

    return profile
