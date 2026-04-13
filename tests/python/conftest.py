import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from helpers import GlmOps, ATOL, RTOL

GPU_ID = int(os.environ.get("GLM_GPU", "0"))


@pytest.fixture(scope="session")
def glm():
    ops = GlmOps(device_id=GPU_ID)
    yield ops
    del ops


@pytest.fixture
def device():
    return torch.device(f"cuda:{GPU_ID}")
