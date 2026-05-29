#!/usr/bin/env python3
"""OpenVINO-first model loading for r2_vision.

Every task model is loaded as OpenVINO IR.  A source path may be:

  • an OpenVINO IR directory          → loaded directly
  • a ``.onnx`` file                  → converted to IR once (openvino.convert_model)
  • a ``.pt`` file                    → exported to IR once (ultralytics export)

Exported IR is cached next to the source using the ultralytics
``<stem>_openvino_model/`` convention, so the conversion runs only on the
first load.
"""

import os

from ultralytics import YOLO


def _ir_dir_for(path):
    stem, _ = os.path.splitext(path)
    return stem + '_openvino_model'


def _export_onnx_to_ir(onnx_path, ir_dir, log):
    """Convert an ONNX model to OpenVINO IR and copy its class-name metadata."""
    import openvino as ov

    os.makedirs(ir_dir, exist_ok=True)
    ov_model = ov.convert_model(onnx_path)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    ov.save_model(ov_model, os.path.join(ir_dir, stem + '.xml'))

    # Carry class names across so YOLO(ir_dir).names works.
    try:
        import onnx as _onnx
        import yaml as _yaml
        meta = {p.key: p.value for p in _onnx.load(onnx_path).metadata_props}
        with open(os.path.join(ir_dir, 'metadata.yaml'), 'w') as f:
            _yaml.dump(meta, f)
    except Exception as e:
        log(f'metadata copy skipped: {e}')


def ensure_openvino_ir(path, log=print):
    """Return a path to an OpenVINO IR directory for ``path``.

    Exports/converts on first call when the source is ``.pt``/``.onnx``.
    Returns ``None`` if the source does not exist (caller treats as abstract).
    """
    if not path:
        return None

    # Already an IR directory.
    if os.path.isdir(path):
        return path

    if not os.path.exists(path):
        return None

    ir_dir = _ir_dir_for(path)
    if os.path.isdir(ir_dir):
        return ir_dir

    ext = os.path.splitext(path)[1].lower()
    if ext == '.onnx':
        log(f'Converting {os.path.basename(path)} → OpenVINO IR ...')
        _export_onnx_to_ir(path, ir_dir, log)
    else:
        log(f'Exporting {os.path.basename(path)} → OpenVINO IR ...')
        YOLO(path).export(format='openvino', imgsz=640)

    return ir_dir if os.path.isdir(ir_dir) else None


def load_model(path, log=print):
    """Load ``path`` as an OpenVINO-backed YOLO, or return ``None`` if absent."""
    ir = ensure_openvino_ir(path, log=log)
    if not ir:
        return None
    log(f'Loading OpenVINO IR: {os.path.basename(ir)}')
    return YOLO(ir)
