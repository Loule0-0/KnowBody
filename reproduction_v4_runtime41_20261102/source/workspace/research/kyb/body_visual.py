"""Project public X5 body geometry into the synchronized, unmodified RGB views."""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def render_intrinsics(resolution, focal_length, horizontal_aperture, *,
                      lens_model, projection_mode, aperture_offsets):
    """Verified Omniverse pinhole contract; integer indices denote pixel centres."""
    width, height = resolution
    values = np.asarray([width, height, focal_length, horizontal_aperture, *aperture_offsets], dtype=float)
    if (values.shape != (6,) or not np.isfinite(values).all()
        or min(values[:4]) <= 0 or width != int(width) or height != int(height)
        or lens_model != 'pinhole' or projection_mode != 'perspective'
        or any(float(x) != 0 for x in aperture_offsets)):
        raise ValueError('Unsupported camera: require perspective pinhole, finite dimensions and zero aperture offsets')
    focal_pixels = width * focal_length / horizontal_aperture
    return np.array([[focal_pixels, 0, (width-1)/2],
                     [0, focal_pixels, (height-1)/2], [0, 0, 1]], dtype=float)


def project(points_world, intrinsics, world_to_optical, resolution):
    """Preserve depth/out-of-frame facts; never certify visibility or contact."""
    points = np.asarray(points_world, dtype=float).reshape(-1, 3)
    k, v = np.asarray(intrinsics, dtype=float), np.asarray(world_to_optical, dtype=float)
    if k.shape != (3, 3) or v.shape != (4, 4) or not all(np.isfinite(x).all() for x in (points, k, v)):
        raise ValueError('Nonfinite or malformed body projection input')
    optical = points @ v[:3, :3].T + v[:3, 3]
    pixels = optical @ k.T
    result = []
    width, height = resolution
    for p, h in zip(optical, pixels):
        uv = h[:2] / h[2] if p[2] > 0 else None
        result.append({'pixel_uv': np.round(uv, 3).tolist() if uv is not None else None,
            'optical_depth_m': round(float(p[2]), 6),
            'in_image': bool(uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height)})
    return result


def flange_transform(pose):
    from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.pose import quat_to_matrix
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-9:
        raise ValueError('Invalid measured flange pose')
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(pose[3:])
    t[:3, 3] = pose[:3]
    return t


class BodyVisualReference:
    """Explicit X5 geometry, measured flange/fingers, actual camera matrices. No object GT."""
    def __init__(self, urdf_path: Path, grasp_offset_m: float):
        self.urdf_path = Path(urdf_path)
        self.grasp_offset = grasp_offset_m
        root = ET.parse(self.urdf_path).getroot()
        self.fingers = []
        sources = {str(self.urdf_path): hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()}
        for number, inward_sign in ((7, -1), (8, 1)):
            joint = root.find(f"joint[@name='joint{number}']")
            assert joint.attrib['type'] == 'prismatic' and joint.find('parent').attrib['link'] == 'link6'
            origin = joint.find('origin')
            t = np.eye(4)
            t[:3, 3] = np.fromstring(origin.attrib['xyz'], sep=' ')
            assert np.allclose(np.fromstring(origin.attrib['rpy'], sep=' '), 0)
            axis = np.fromstring(joint.find('axis').attrib['xyz'], sep=' ')
            visual = root.find(f"link[@name='link{number}']/visual")
            assert np.allclose(np.fromstring(visual.find('origin').attrib['xyz'], sep=' '), 0)
            assert np.allclose(np.fromstring(visual.find('origin').attrib['rpy'], sep=' '), 0)
            path = self.urdf_path.parent / visual.find('geometry/mesh').attrib['filename']
            raw = path.read_bytes()
            count = struct.unpack_from('<I', raw, 80)[0]
            if len(raw) != 84 + 50 * count:
                raise ValueError('Expected the pinned binary X5 finger STL')
            triangles = np.frombuffer(raw, offset=84, count=count,
                dtype=np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]))
            vertices, normals = triangles['vertices'].astype(float), triangles['normal'].astype(float)
            tip = vertices[:, :, 0].max()
            inner_y = vertices[:, :, 1].min() if inward_sign < 0 else vertices[:, :, 1].max()
            selected = ((normals[:, 1] * inward_sign > .999)
                & (np.abs(vertices[:, :, 1] - inner_y).max(axis=1) < 1e-7)
                & (vertices[:, :, 0].min(axis=1) >= tip - .0151))
            face = vertices[selected]
            if len(face) != 2:
                raise ValueError('Pinned X5 distal inner face selection changed')
            areas = np.linalg.norm(np.cross(face[:, 1]-face[:, 0], face[:, 2]-face[:, 0]), axis=1) / 2
            centroid = np.average(face.mean(axis=1), axis=0, weights=areas)
            self.fingers.append((t, axis, face, centroid))
            sources[str(path)] = hashlib.sha256(raw).hexdigest()
        self.sources = sources

    def inner_face_windows(self, measured_fingers):
        """Actual public distal pad outline in flange X/Z; fingers translate along Y."""
        result = {}
        for arm in ('left', 'right'):
            windows = {}
            for number, ((joint_t, axis, face, _), opening) in enumerate(zip(self.fingers, measured_fingers[arm]), 7):
                points = face.reshape(-1, 3) + joint_t[:3, 3] + axis * opening
                polygon = np.unique(points[:, [0, 2]], axis=0)
                center = polygon.mean(axis=0)
                order = np.argsort(np.arctan2(polygon[:, 1]-center[1], polygon[:, 0]-center[0]))
                windows[f'finger{number}'] = {'polygon_xz_m': polygon[order].tolist(),
                    'plane_y_m': float(points[0, 1]), 'inward_y_sign': -1 if number == 7 else 1}
            result[arm] = windows
        return result

    def geometry(self, state, measured_fingers):
        result = {}
        for arm in ('left', 'right'):
            t = flange_transform(state[f'{arm}_ee_pose'])
            points = {'named_grasp': (t @ [self.grasp_offset, 0, 0, 1])[:3]}
            for i, ((joint_t, axis, face, centroid), opening) in enumerate(zip(self.fingers, measured_fingers[arm])):
                finger_t = joint_t.copy()
                finger_t[:3, 3] += joint_t[:3, :3] @ (axis * opening)
                world = t @ finger_t
                points[f'finger{i+7}_inner_face_triangles'] = face.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
                points[f'finger{i+7}_inner_face_centroid'] = (world @ [*centroid, 1])[:3]
            result[arm] = points
        return result

    def capture(self, camera_manager, state, measured_fingers, images):
        geometry = self.geometry(state, measured_fingers)
        views = {}
        for name, camera in zip(camera_manager.camera_names[0], camera_manager.cameras[0]):
            if name not in images:
                continue
            from pxr import UsdGeom
            usd_camera = UsdGeom.Camera(camera.prim)
            offsets = [usd_camera.GetHorizontalApertureOffsetAttr().Get(),
                       usd_camera.GetVerticalApertureOffsetAttr().Get()]
            sdk_k = np.asarray(camera.get_intrinsics_matrix(device='cpu'), dtype=float)
            v = np.asarray(camera.get_view_matrix_ros(device='cpu'), dtype=float)
            resolution = camera.get_resolution()
            k = render_intrinsics(resolution, camera.get_focal_length(), camera.get_horizontal_aperture(),
                lens_model=camera.get_lens_distortion_model(), projection_mode=camera.get_projection_mode(),
                aperture_offsets=offsets)
            bodies = {}
            for arm, points in geometry.items():
                bodies[arm] = {key: project(value, k, v, resolution) for key, value in points.items()}
            view = {'image_sha256': images[name]['sha256'], 'resolution_wh': list(resolution),
                'intrinsics_K': k.tolist(), 'sdk_intrinsics_K_diagnostic_only': sdk_k.tolist(),
                'projection_contract': 'Omniverse perspective pinhole horizontal_FOV square_pixels; integer_uv_pixel_centres',
                'aperture_offsets': offsets,
                'world_to_optical_ros': v.tolist(), 'body_projections': bodies}
            arm = next((a for a in geometry if name == f'cam_{a}_wrist'), None)
            if arm:
                fixed_point = geometry[arm]['named_grasp']
                baseline = project(fixed_point, k, v, resolution)[0]
                if baseline['pixel_uv'] is not None:
                    gains = {}
                    for axis, delta in zip('xyz', np.eye(3) * .001):
                        shifted_v = v.copy()
                        shifted_v[:3, 3] -= v[:3, :3] @ delta
                        shifted = project(fixed_point, k, shifted_v, resolution)[0]
                        gains[axis] = (np.asarray(shifted['pixel_uv']) - baseline['pixel_uv']).round(3).tolist()
                    view['reference_scene_point_pixel_change_for_world_positive_1mm_camera_translation'] = {
                        'delta_uv': gains, 'reference_optical_depth_m': baseline['optical_depth_m'],
                        'assumptions': 'Stationary scene point at this nominal grasp depth; camera orientation fixed. '
                            'Unknown tool depth changes the gain. Own grasp co-moves with its wrist camera and has zero translation response.'}
            views[name] = view
        if set(views) != set(images):
            raise ValueError('RGB and body camera reference sets differ')
        return {'source': 'public_robot_geometry_and_current_measured_body_and_camera_matrices',
            'geometry_sources_sha256': self.sources, 'grasp_offset_body_x_m': self.grasp_offset,
            'inner_face_windows_flange_m': self.inner_face_windows(measured_fingers),
            'nominal_inner_face_separation_m': {arm:float(
                self.fingers[0][0][1,3]+self.fingers[0][3][1]+measured_fingers[arm][0]
                -self.fingers[1][0][1,3]-self.fingers[1][3][1]+measured_fingers[arm][1]) for arm in geometry},
            'occlusion': 'unknown', 'interpretation': 'Pixel origins are top-left; u right, v down. '
                'Named grasp, distal inner finger faces, and unknown tool contact point are different. '
                'Projected triangles describe body surfaces, not a valid grasp volume or contact/retention proof. '
                'No object pose, depth, contact, or evaluation state is used.',
            'views': views}
