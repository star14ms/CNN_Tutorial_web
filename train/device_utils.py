"""
device_utils.py — Cross-platform device detection for PyTorch & ONNX Runtime

Automatically detects and selects the best available device:
- NVIDIA CUDA (NVIDIA GPUs)
- Apple Metal Performance Shaders (Apple Silicon Macs)
- CPU fallback

Works with both PyTorch and ONNX Runtime.
"""

import torch

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


def get_device() -> torch.device:
    """
    Get the best available device for training.

    Priority:
    1. CUDA (NVIDIA GPU) - if available
    2. MPS (Apple Silicon) - if available and functional
    3. CPU - fallback

    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS is available on Apple Silicon
        device = torch.device('mps')
        device_name = "Apple Metal Performance Shaders (MPS)"
    else:
        device = torch.device('cpu')
        device_name = "CPU"

    return device


def get_device_with_info() -> tuple:
    """
    Get device and print detailed information.

    Returns:
        tuple: (torch.device, device_name_string)
    """
    device = get_device()

    if device.type == 'cuda':
        device_str = f'CUDA ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)'
    elif device.type == 'mps':
        device_str = 'Apple MPS (Metal Performance Shaders)'
    else:
        device_str = 'CPU'

    return device, device_str


# Backward compatibility - some scripts might import get_device_name
def get_device_name() -> str:
    """Get a human-readable device name."""
    _, name = get_device_with_info()
    return name


def get_onnx_providers() -> list:
    """
    Get optimal ONNX Runtime execution providers.

    Priority:
    1. CUDAExecutionProvider (NVIDIA GPU)
    2. CoreMLExecutionProvider (Apple Metal - for M1/M2/M3)
    3. CPUExecutionProvider (CPU fallback)

    Returns:
        list: List of provider names to pass to ort.InferenceSession

    Example:
        sess = ort.InferenceSession(model_path, providers=get_onnx_providers())
    """
    if not HAS_ONNXRUNTIME:
        return []

    available_providers = ort.get_available_providers()

    # Try CUDA first (NVIDIA GPU)
    if 'CUDAExecutionProvider' in available_providers:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # Try CoreML (Apple Silicon - uses Metal)
    if 'CoreMLExecutionProvider' in available_providers:
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']

    # Fallback to CPU
    return ['CPUExecutionProvider']


def get_onnx_providers_with_info() -> tuple:
    """
    Get ONNX providers and print information.

    Returns:
        tuple: (list of providers, info_string)
    """
    providers = get_onnx_providers()

    if not providers:
        return [], "ONNX Runtime not available"

    provider_names = {
        'CUDAExecutionProvider': 'NVIDIA CUDA',
        'CoreMLExecutionProvider': 'Apple Metal (CoreML)',
        'CPUExecutionProvider': 'CPU',
    }

    primary = providers[0]
    info_str = provider_names.get(primary, primary)

    return providers, info_str
