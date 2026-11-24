import os
import pickle

import numpy as np
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_ALLOWED_PREFIXES = ("torch", "collections", "numpy", "_codecs", "argparse")
_ALLOWED_BUILTINS = {"set", "dict", "list", "tuple", "int", "float", "str", "bool", "complex", "slice", "bytearray"}

_mean = np.array(MEAN, np.float32)[:, None, None]
_std = np.array(STD, np.float32)[:, None, None]


def to_tensor(img):
    a = np.asarray(img.convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((a - _mean) / _std)


def _find_unzipped(path):
    """Return the folder that holds data.pkl (an UNZIPPED torch checkpoint), or None."""
    for root, _, files in os.walk(path):
        if "data.pkl" in files:
            return root
    return None


def resolve_checkpoint(path):
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint not found: {path}")
    if os.path.isdir(path):
        d = _find_unzipped(path)
        if d:
            print(f"[ckpt] '{path}' is an unzipped checkpoint folder; reading it directly from {d}")
            return d
        found = [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs
                 if f.lower().endswith((".pth", ".pt", ".tar", ".bin"))]
        if len(found) != 1:
            raise SystemExit(f"'{path}' is a FOLDER, not a file, and has no checkpoint I recognise. "
                             f"Contents: {os.listdir(path)}")
        print(f"[ckpt] '{path}' is a folder; using the file inside: {found[0]}")
        return found[0]
    return path


def load_unzipped(d):
    """Load a torch checkpoint that was extracted from its zip container (data.pkl + data/<n> + version).
    Tensor files are memory-mapped, so the 8 GB are not read into RAM up front.
    Unpickling is restricted to torch / collections / numpy classes."""
    data_dir = os.path.join(d, "data")

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(_ALLOWED_PREFIXES) or (module == "builtins" and name in _ALLOWED_BUILTINS):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")

        def persistent_load(self, pid):
            typ, storage_type, key, _location, numel = pid[:5]
            if typ != "storage":
                raise pickle.UnpicklingError(f"unknown persistent id {typ}")
            dtype = storage_type.dtype
            nbytes = numel * torch._utils._element_size(dtype)
            us = torch.UntypedStorage.from_file(os.path.join(data_dir, str(key)), False, nbytes)
            return torch.storage.TypedStorage(wrap_storage=us, dtype=dtype, _internal=True)

    with open(os.path.join(d, "data.pkl"), "rb") as f:
        return _U(f).load()


def load_state(path):
    """Load a (possibly 8 GB) checkpoint with as little RAM as possible."""
    path = resolve_checkpoint(path)
    if os.path.isdir(path):
        return load_unzipped(path)
    try:  # safest + memory mapped
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as e:  # older format or non-tensor objects inside
        print(f"[info] safe/mmap load failed ({type(e).__name__}); falling back to a normal load. "
              f"Only do this for files you downloaded from the official source.")
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(ckpt):
    sd = ckpt
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
    return {k.replace("module.", "", 1): v for k, v in sd.items() if hasattr(v, "shape")}


def build_model(model_name, num_classes, checkpoint=None):
    from fastervit import create_model

    try:
        model = create_model(model_name, pretrained=False, num_classes=num_classes)
    except Exception as e:
        try:
            import timm
            avail = timm.list_models("faster_vit*")
        except Exception:
            avail = []
        raise SystemExit(f"Could not create '{model_name}': {e}\nAvailable FasterViT names: {avail}")

    if checkpoint:
        sd = extract_state_dict(load_state(checkpoint))
        msd = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        skipped = [k for k in sd if k not in keep]
        model.load_state_dict(keep, strict=False)
        print(f"[ckpt] loaded {len(keep)}/{len(msd)} tensors, skipped {len(skipped)} "
              f"(expected: the old 21k/1k classifier head). First skipped: {skipped[:4]}")
        if len(keep) < 0.9 * len(msd):
            print("[WARN] many tensors did not match - check --model-name / --img-size vs. the checkpoint!")
    return model