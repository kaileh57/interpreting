"""Shared helpers for RCG-Bench notebooks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# NOTE: torch is imported lazily inside functions so this module imports even in
# a bare kernel; colab_bootstrap() can then pip-install deps before torch is used.


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def repo_root() -> Path:
    cwd = Path.cwd()
    for candidate in (cwd, cwd.parent):
        if (candidate / "src" / "rcg").exists():
            return candidate
    return cwd


def load_hf_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if "google.colab" in sys.modules:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        return token
    env_file = repo_root() / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["HF_TOKEN"] = token
                return token
    return None


def hf_login() -> None:
    token = load_hf_token()
    if not token:
        raise RuntimeError("Set Colab secret HF_TOKEN (read token). See README.")
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def _needs_install() -> bool:
    """Install when running in Colab or when the package/deps aren't importable."""
    if os.environ.get("RCG_SKIP_INSTALL"):
        return False
    if "google.colab" in sys.modules:
        return True
    try:
        import pandas  # noqa: F401
        import transformers  # noqa: F401
        return False
    except Exception:
        return True


def colab_bootstrap(install: bool = True, require_gpu: bool = True) -> Path:
    root = repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    if load_hf_token():
        hf_login()

    if install and _needs_install():
        extras = "[dev,sae]"
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-e", f"{root}{extras}"]
        )

    if not (root / "data" / "latent_slot_tasks" / "dataset.json").exists():
        subprocess.check_call([sys.executable, "scripts/generate_datasets.py"], cwd=root)

    if require_gpu and "google.colab" in sys.modules and not _cuda_available():
        raise RuntimeError("Enable GPU: Runtime → Change runtime type → GPU")

    return root


def pick_model_id(preferred: str = "google/gemma-3-270m-it", fallback: str = "gpt2") -> str:
    if os.environ.get("RCG_MODEL_ID"):
        return os.environ["RCG_MODEL_ID"]
    return preferred if _cuda_available() else fallback


def gpu_banner(preferred: str = "google/gemma-3-270m-it") -> str:
    import torch

    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "CPU"
    hf = "ok" if os.environ.get("HF_TOKEN") else "missing"
    return f"{name} | {pick_model_id(preferred)} | HF: {hf}"
