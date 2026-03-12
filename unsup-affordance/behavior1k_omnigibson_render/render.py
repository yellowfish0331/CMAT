import os
import random
import numpy as np
import json
import pickle
import copy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import h5py

import omnigibson as og
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.objects import DatasetObject
from omnigibson.sensors import VisionSensor
from omnigibson.utils.transform_utils import pose2mat, pose_inv
from omnigibson.object_states import ToggledOn

from omnigibson.macros import gm

import torch as th


gm.ENABLE_FLATCACHE = True

def create_empty_scene():
    """Create an empty scene with a default camera"""
    cfg = dict()
    cfg["scene"] = {
        "type": "Scene", 
        "floor_plane_visible": False,
    }
    env = og.Environment(cfg)
    og.sim.enable_viewer_camera_teleoperation()

    cam = og.sim.viewer_camera
    cam.image_height = 784
    cam.image_width = 784
    for mod in ["rgb", "depth_linear", "seg_instance", "seg_semantic", "seg_instance_id"]:
        cam.add_modality(mod)

    return cam

def create_dataset_object(category, **kwargs):
    """Create a dataset object with default setup"""

    if 'model' not in kwargs:
        kwargs['model'] = random.choice(get_all_object_category_models(category))

    # randomize idx to avoid name collision
    if 'idx' not in kwargs:
        kwargs['idx'] = random.randint(0, 10)

    name = f"{category}_{kwargs['model']}_{kwargs['idx']}"
    
    obj = DatasetObject(
        prim_path=f"/World/{name}",
        name=name,
        category=category,
        fit_avg_dim_volume=True,
        **kwargs
    )

    return obj

def import_obj(obj):
    """Import an object into the scene at origin"""
    og.sim.stop()
    og.sim.batch_add_objects([obj], scenes=og.sim.scenes)
    og.sim.play()
    for _ in range(10): og.sim.step()
    print(f"Imported {obj.name} with location and orientation {obj.get_position_orientation()}")

def remove_obj(obj):
    """Remove an object from the scene"""
    og.sim.stop()
    og.sim.batch_remove_objects([obj])
    og.sim.play()
    for _ in range(10): og.sim.step()
    print(f"Removed {obj.name}")

def get_object_metadata(obj):
    """Get metadata of an object"""
    # convert to json-serializable format
    pose = obj.get_position_orientation()
    pose_list = [item.tolist() for item in pose]
    scale = obj.aabb_extent.tolist()

    metadata = {
        'category': obj.category,
        'model': obj.model,
        'pose': pose_list,
        'scale': scale,
    }
    return metadata

def compute_camera_intrinsics(cam):
    """Compute camera intrinsics, assuming square image"""
    img_width = cam.image_width
    img_height = cam.image_height
    apert = cam.prim.GetAttribute("horizontalAperture").Get()
    focal_len_in_pixel = cam.focal_length * img_width / apert

    intrinsics = np.eye(3)
    intrinsics[0,0] = focal_len_in_pixel
    intrinsics[1,1] = focal_len_in_pixel
    intrinsics[0,2] = img_width / 2
    intrinsics[1,2] = img_height / 2

    return intrinsics

def set_camera_view_at_obj(cam, ref_obj, pitch, yaw, dist):
    """Set camera viewing ref_obj with specified pose. Returns camera pose."""

    from omni.isaac.core.utils.viewports import set_camera_view
    
    # ref_pos = ref_obj.get_position()
    ref_pos = ref_obj.get_position_orientation()[0]
    target_to_cam = R.from_euler("yz", [pitch, yaw]).apply([1, 0, 0])
    raw_cam_pos = target_to_cam * dist
    cam_pos = ref_pos + raw_cam_pos
    set_camera_view(cam_pos, ref_pos, camera_prim_path="/World/viewer_camera", viewport_api=None)
    for _ in range(10): og.sim.render()

    return cam.get_position_orientation()

def create_scan(cam, ref_obj, pitch_list=[-np.pi/4, 0, np.pi/4], dist=None, n_interp=10):
    """Create a scan of ref_obj with camera rotateing at specified pitches"""
    def rotate_around_obj(pitch):
        poses = []
        for yaw in np.linspace(0, 2*np.pi, n_interp):
            pose = set_camera_view_at_obj(cam, ref_obj, pitch, yaw, dist)
            poses.append(pose)
        return poses
    
    # adaptively set dist
    if dist is None:
        dist = ref_obj.aabb_extent.max().item() * 1.8

    poses = []
    for pitch in pitch_list:
        # ignore last pose since it is the same as the first
        poses += rotate_around_obj(pitch=pitch)[:-1]

    metadata = {'num_poses': len(poses), 'ref_obj': ref_obj.name,
                'pitch_list': pitch_list, 'dist': dist, 'n_interp': n_interp}

    # return the metadata of parameters actually used
    return poses, metadata

def render_dataset_object(obj, cam, path, **kwargs):
    '''
    Creates a scan of the dataset object and stores relevant information.
    Input:
        obj: DatasetObject
        cam: Camera             ----- assume already created in scene
        path: str               ----- path to store the scan

    '''
    # collect camera poses for the scan
    poses, scan_metadata = create_scan(cam, obj, **kwargs)

    # collect metadata of the object, store in json
    obj_metadata = get_object_metadata(obj)

    # render the scan, store image, depth, mask
    scan_metadata['cam_intrinsic'] = compute_camera_intrinsics(cam).tolist()

    metadata = {"object": obj_metadata, "scan": scan_metadata}
    with open(f"{path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)

    scan = {"cam_pose": [], "cam_extrinsic": [], "rgb": [], "depth": [], "link_seg": []}    
    for i, pose in enumerate(poses):
        # set camera pose and render
        cam.set_position_orientation(position=pose[0], orientation=pose[1])
        for _ in range(10): og.sim.render()

        scan['cam_pose'].append(np.concatenate(pose))
        scan['cam_extrinsic'].append(pose_inv(pose2mat(pose)))

        obs = cam.get_obs()[0]
        rgb = obs['rgb'][:, :, :3]
        if type(obs['seg_instance']) == tuple:
            mask = obs['seg_instance'][0]
        else:
            mask = obs['seg_instance']
        if mask.shape != rgb.shape[:2]:
            print("Mask shape does not match rgb shape")
            return
        mask = mask.cpu()
        rgb[mask == 0] = th.tensor([255, 255, 255], dtype=th.uint8) # set background to white
        depth = obs['depth_linear']
        depth[mask == 0] = 0. # use 0 to represent background
        link_seg = obs['seg_instance_id']
        link_seg[mask == 0] = 0

        scan['rgb'].append(copy.copy(rgb))
        scan['depth'].append(copy.copy(depth))
        scan['link_seg'].append(copy.copy(link_seg))

        if i == 0: # save a sample image
            # plt.imsave(f"{path}/rgb_{i}.png", obs['rgb'])
            rgb_image_to_save = obs['rgb'].cpu().numpy().astype(np.uint8)
            plt.imsave(f"{path}/rgb_{i}.png", rgb_image_to_save)

    save_scan_to_h5(scan, f"{path}/scan.h5")

def save_scan_to_h5(scan, path, conpression_level=5):
    """Save the scan to h5 file"""
    with h5py.File(path, 'w') as f:
        for key, value in scan.items():
            if th.is_tensor(value[0]):
                # convert tensor to numpy array
                value = [v.cpu().numpy() for v in value]
            value = np.stack(value, axis=0)
            f.create_dataset(key, data=value, compression="gzip", compression_opts=conpression_level)

def load_scan_from_h5(path):
    """Load the scan from h5 file"""
    h5_file = h5py.File(f"{path}/scan.h5", 'r')
    scan = {"frames":{}}
    json_file = json.load(open(f"{path}/metadata.json", 'r'))
    scan['metadata'] = json_file['scan']

    num_frames = h5_file['cam_pose'].shape[0]
    for i in range(num_frames):
        scan['frames'][i] = {}
        scan['frames'][i]['cam_pose'] = (h5_file['cam_pose'][i][:3], h5_file['cam_pose'][i][3:])
        scan['frames'][i]['cam_extrinsic'] = h5_file['cam_extrinsic'][i]
        scan['frames'][i]['rgb'] = h5_file['rgb'][i]
        scan['frames'][i]['depth'] = h5_file['depth'][i]
        scan['frames'][i]['link_seg'] = h5_file['link_seg'][i]
        scan['frames'][i]['mask'] = (h5_file['depth'][i] != 0).astype(np.uint8)

    return scan

def render_pipeline_category_model(category_model_list, args):
    """Renders random objects from the specified categories"""
    cam = create_empty_scene()

    for idx, d in enumerate(category_model_list):
        category = d['category']
        model = d['model']

        print(f"Rendering category {category} model {model}")

        # check orientation file first to ensure category-model exists
        orientation_fpath = f"{args.orientation_root}/{category}/{model}.json"
        print(f"Orientation file: {orientation_fpath}")
        if not os.path.exists(orientation_fpath):
            print(f"Orientation file for {category} {model} does not exist, skipping")
            continue

        if not os.path.exists(os.path.join(args.og_dataset_root, category, model)):
            print(f"Object {category} {model} does not exist in og_dataset, skipping")
            continue
        os.makedirs(f"{args.save_path}/og_raw/{category}", exist_ok=True)

        path = f"{args.save_path}/og_raw/{category}/{model}"
        os.makedirs(path, exist_ok=True)

        obj = create_dataset_object(category, model=model, idx=idx)
        import_obj(obj)

        # disable visual marker for toggle-able objects
        if ToggledOn in obj.states:
            print(f"Disabling visual marker for {obj.name}")
            obj.states[ToggledOn].visual_marker.visible = False
            for _ in range(10): og.sim.step()

        orientation_fpath = f"{args.orientation_root}/{category}/{model}.json"
        orientation_f = json.load(open(orientation_fpath, 'r'))
        # obj.set_orientation(orientation_f['orientation'])
        obj.set_position_orientation(orientation=orientation_f['orientation'])
        print(f"Set orientation to {orientation_f['orientation']}")
        for _ in range(10): og.sim.step()

        render_dataset_object(obj, cam, path, pitch_list=[-np.pi/3, -np.pi/6, 0, np.pi/8], n_interp=11)

        remove_obj(obj)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--orientation_root", type=str, required=True)
    parser.add_argument("--og_dataset_root", type=str, required=True)
    parser.add_argument("--category_model_list", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    category_model_list = json.load(open(args.category_model_list, 'r'))
    render_pipeline_category_model(category_model_list, args)

