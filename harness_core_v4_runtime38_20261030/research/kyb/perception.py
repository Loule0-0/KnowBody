"""The observation boundary is selected once, before collecting any evidence."""
from __future__ import annotations
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Perception:
    name: str
    depth: bool
    calibrated_camera: bool

    @property
    def forbidden_tools(self):
        return frozenset() if self.depth else frozenset(('inspect_surface', 'read_body'))

    def check_memory(self, store):
        if self.depth:
            return
        # Fail before any inference. An empty new chain or this exact profile's
        # previous episodes are allowed; hiding privileged tools is insufficient.
        configured = set()
        for row in store.db.execute("SELECT run,payload FROM evidence WHERE kind='known_config'"):
            data = json.loads(row['payload']); configured.add(row['run'])
            if data.get('perception_profile') != self.name:
                raise ValueError('RGB-only requires an independent, matching-profile memory chain')
        observed = {r['run'] for r in store.db.execute("SELECT DISTINCT run FROM evidence WHERE kind='observation'")}
        if observed - configured:
            raise ValueError('RGB-only historical observations require a matching source configuration')
        # SQL selects suspect payloads without decoding the entire growing raw
        # trajectory archive into Python. Apply the same guard as new writes.
        for row in store.db.execute("""SELECT kind,payload FROM evidence WHERE kind IN ('known_config','surface_inspection')
            OR json_type(payload,'$.depth_snapshots') IS NOT NULL
            OR json_type(payload,'$.body_visual_reference') IS NOT NULL
            OR json_type(payload,'$.actual_RGBD_camera_config') IS NOT NULL
            OR json_type(payload,'$.RGB_camera_identity') IS NOT NULL
            OR json_type(payload,'$.depth_protocol') IS NOT NULL"""):
            self.check_payload(row['kind'], json.loads(row['payload']))

    def check_payload(self, kind, data):
        if self.depth:
            return
        if any(data.get(k) for k in ('depth_snapshots', 'body_visual_reference',
            'actual_RGBD_camera_config', 'RGB_camera_identity', 'depth_protocol')):
            raise ValueError('Privileged camera/depth data violates the RGB-only observation contract')
        if kind == 'surface_inspection':
            raise ValueError('Depth-derived evidence is unavailable in RGB-only mode')


RGBD_CALIBRATED = Perception('rgbd_calibrated', True, True)
RGB_ONLY = Perception('rgb_only', False, False)
PROFILES = {p.name: p for p in (RGBD_CALIBRATED, RGB_ONLY)}


def select(name):
    if name not in PROFILES:
        raise ValueError('Explicit perception_profile must be rgbd_calibrated or rgb_only')
    return PROFILES[name]
