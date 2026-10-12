"""Adapters turn source files into Level 1 detections. Pick one with `inputs.adapter`."""
from importlib import import_module


def get_adapter(name: str):
    return import_module(f"{__name__}.{name}")
