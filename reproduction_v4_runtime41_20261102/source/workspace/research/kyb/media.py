"""Read and render registered JPEGs without granting reader or writer authority."""
import base64
import hashlib
import json
from pathlib import Path


def read_rgb(images, cameras):
    result = []
    for camera in cameras:
        if camera not in images:
            raise ValueError('Unknown camera ' + camera)
        ref = images[camera]
        raw = Path(ref['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref['sha256']:
            raise ValueError('Historical RGB changed')
        result.append((camera, raw))
    return result


def rgb_reference(reference, *, temporal_role):
    return 'RGB_REFERENCE ' + json.dumps(
        {**reference, 'temporal_role': temporal_role}, ensure_ascii=False)


def rgb_parts(frames, *, temporal_role):
    parts = []
    for reference, raw in frames:
        parts.extend([
            {'type': 'text', 'text': rgb_reference(reference, temporal_role=temporal_role)},
            {'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()}}])
    return parts
