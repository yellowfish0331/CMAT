"""
The main class to handle 3D feature fusion and clustering.
3D fusion code adapted from D3Fields https://github.com/WangYixuan12/d3fields
"""

import numpy as np
import os
import sys
import cv2
import torch
import copy
import open3d as o3d
import pickle
from scipy.spatial import cKDTree
import h5py

from utils.pcd_utils import (
    project_points_coords,
    aggr_point_cloud_from_data, normalize_pointcloud
    )
from utils.img_utils import (
    get_dino_features,
    get_palette
    )

#######################################
## Fusion class
#######################################
class Fusion:
    def __init__(self, pcd, data, rescale_params=None):
        """
        Initialize the Fusion class with input point cloud and raw data.
        Input:
            pcd: open3d.geometry.PointCloud, input point cloud
            data: dict, raw data including colors, depths, intrinsics, extrinsics, features
        """
        # input pcd and data
        self.points = np.asarray(pcd.points) # [N, 3]
        self.colors = np.asarray(pcd.colors) # [N, 3]

        # if pcd is rescaled, need orig_pts = rescaled_pts / scale - translation
        self.rescale_params = rescale_params 

        # raw data 
        self.images = data['colors'] # [num_cam, H, W, C]
        self.depths = data['depths'] # [num_cam, H, W]
        self.link_segs = data['link_segs'].astype(np.int8) # [num_cam, H, W]
        self.intrinsics = data['intrinsics'] # [num_cam, 3, 3]
        self.extrinsics = data['extrinsics'] # [num_cam, 4, 4]
        self.raw_features = data['features'] # [num_cam, H, W, feat_dim]

        self.num_frames = self.images.shape[0]

        # fused features
        self.mu = 0.002 # point valid (on surface) if |depth_reading - depth_proj| <= mu
        self.fused_weighted_features = None # [N, feat_dim]
        self.fused_weights = np.zeros(self.points.shape[0]) # [N, ]

        # fused link segs
        self.projected_link_segs = np.full((self.points.shape[0], self.images.shape[0]), -1, dtype=np.int8) # [N, num_cam]

        self.fused_frames = set()

        # find projection
        self.find_proj_idx()

    #######################################
    ## Fusing features to point cloud
    #######################################

    # TODO: currently directly adapting implementation from D3Fields. Could be optimized.
    def fuse_frame(self, frame_idx):
        """
        Fuse the features of a single frame to pointcloud.
        Update self.fused_weighted_features and self.fused_weights.
        """
        if frame_idx in self.fused_frames:
            print(f"frame {frame_idx} has been fused already, skipped")
            return 
        
        self.fused_frames.add(frame_idx)
        
        pts = torch.tensor(self.points)
        Rt = torch.tensor(self.extrinsics)[:,:3, :]
        K = torch.tensor(self.intrinsics)

        pts_2d, valid_mask, depth = project_points_coords(pts, Rt, K, rescale_params=self.rescale_params)
        pts_2d = pts_2d.cpu().numpy().astype(int)[frame_idx]
        valid_mask = valid_mask.cpu().numpy()[frame_idx]
        depth_proj = depth.cpu().numpy()[frame_idx]

        depth_img = self.depths[frame_idx]
        feature_img = self.raw_features[frame_idx]
        seg_img = self.link_segs[frame_idx]

        output = self.process_frame_for_fusion(pts_2d, valid_mask, depth_proj, depth_img, feature_img, seg_img)

        weighted_features = output["weighted_features"]
        valid_mask = output["valid_mask"]

        # fuse features
        if self.fused_weighted_features is None:
            self.fused_weighted_features = weighted_features
        else:
            self.fused_weighted_features = (self.fused_weighted_features * self.fused_weights.reshape(-1, 1) + weighted_features * output["valid_mask"].reshape(-1, 1).astype(float)) / (self.fused_weights.reshape(-1, 1) + output["valid_mask"].astype(float).reshape(-1, 1) + 1e-6)
        self.fused_weights += output["valid_mask"].astype(float)
        self.fused_weighted_features[self.fused_weights == 0] = 0.0

        # fuse link segs
        self.projected_link_segs[:, frame_idx] = output['link_segs'].reshape(1, -1)

        # HACK for now to evaluate the fusion
        return output 

    def process_frame_for_fusion(self, pts_2d, valid_mask, depth_proj, depth_img, feature_img, seg_img=None):
        """
        For given frame, find weights and fuse features to the pointcloud
        """
        # get depth reading
        rows, cols = pts_2d[:,1], pts_2d[:,0]
        # clamp rows and cols to be within the image
        rows = np.clip(rows, 0, depth_img.shape[0]-1)
        cols = np.clip(cols, 0, depth_img.shape[1]-1)
        depth_reading = depth_img[rows, cols]

        # calculate and process distance
        dist = depth_reading - depth_proj.squeeze()
        dist_valid_mask = (depth_reading > 0) & (np.abs(dist) <= self.mu) & valid_mask

        dist_weight = np.exp(np.clip(self.mu - np.abs(dist), a_min=None, a_max=0) / self.mu)

        dist = np.clip(dist, a_min=-self.mu, a_max=self.mu)
        weighted_dist = (dist * dist_valid_mask.astype(float)) / (dist_valid_mask.astype(float) + 1e-6)
        weighted_dist[~dist_valid_mask] = 1e3

        output = {"dist": weighted_dist, "valid_mask": dist_valid_mask, "dist_weight": dist_weight}

        # fuse features
        features = feature_img[rows, cols]
        weighted_features = features
        weighted_features[~dist_valid_mask] = 0.0
        output["weighted_features"] = weighted_features

        # fuse link_segs
        link_segs = seg_img[rows, cols]
        link_segs[~dist_valid_mask] = -1
        output["link_segs"] = link_segs

        return output
    
    @property
    def fused_link_segs(self):
        """
        Finds link label for each 3D point.
        return: [N,] array, link label for each point. -1 for no label.
        """
        print("Fusing link segs...")
        fused_link_segs = np.full(self.points.shape[0], -1, dtype=np.int8)
        max_count = np.zeros(self.points.shape[0], dtype=np.int16)  # count the current max count of any label to each point

        for label in np.unique(self.projected_link_segs):
            if label == -1 or label == 0: # some data has 0 as background label
                continue
            mask = (self.projected_link_segs == label)
            count = np.sum(mask, axis=1)
            update = count > max_count
            fused_link_segs[update] = label
            max_count[update] = count[update]

        return fused_link_segs
    
    #######################################
    ## Projecting pointcloud to image
    #######################################
    def project_points_to_cam(self, frame_idx):
        """
        (separated from original view_projection_to_cam)
        Project the points to the camera view of a given frame.
        Returns valid mask and RAW 2D points.
        """
        pts = torch.tensor(self.points)
        Rt = torch.tensor(self.extrinsics)[:,:3, :]
        K = torch.tensor(self.intrinsics)

        pts_2d, valid_mask, depth = project_points_coords(pts, Rt, K, rescale_params=self.rescale_params)
        pts_2d = pts_2d.cpu().numpy().astype(int)[frame_idx]
        valid_mask = valid_mask.cpu().numpy()[frame_idx]
        depth_proj = depth.cpu().numpy()[frame_idx]

        depth_img = self.depths[frame_idx]
        # get depth reading
        rows, cols = pts_2d[:,1], pts_2d[:,0]
        # clamp rows and cols to be within the image
        rows = np.clip(rows, 0, depth_img.shape[0]-1)
        cols = np.clip(cols, 0, depth_img.shape[1]-1)
        depth_reading = depth_img[rows, cols]
        
        dist = depth_reading - depth_proj.squeeze()
        valid_mask = (depth_reading > 0) & (np.abs(dist) <= self.mu) & valid_mask

        return pts_2d, valid_mask
    
    def view_projection_to_cam(self, frame_idx, labels=None, overlay_orig=0., color_patch = 3):
        """
        Visualize the projection of self.points in a given frame's camera view.
        labels: if provided, use as color labels for points. Assumes colors has been normalized to [0, 1].
                if None, will plot black points on the image.
        overlay_orig: if > 0, overlay the original image with the projected points.
        color_patch: color n*n pixels around the projected point. If 1, will plot a single pixel.

        TODOs:
        - when label not passed in, use self.colors if available
        - update overlay_orig to be the weighted sum of the original image and the projected points
        """
        pts_2d, valid_mask = self.project_points_to_cam(frame_idx)
        rows, cols = pts_2d[:,1], pts_2d[:,0]

        if labels is not None:
            if labels.max() > 1: 
                labels = (labels - labels.min()) / (labels.max() - labels.min())
            if len(labels) != len(pts_2d):
                raise ValueError("labels should have the same length as pts_2d")
            # make label 3 dim
            missing_dim = 3 - labels.shape[-1]
            if missing_dim > 0:
                labels = np.concatenate([labels, np.ones((labels.shape[0], missing_dim))], axis=-1)

        # plot the points
        if overlay_orig > 0:
            img = copy.deepcopy(self.images[frame_idx]) / 255.
            img = img * 0.4 + 0.6 # whiten
        else:
            img = np.ones_like(self.images[frame_idx], dtype=np.float32)
        for i, (r, c) in enumerate(zip(rows, cols)):
            if valid_mask[i]:
                # img[r, c] = labels[i] if labels is not None else [0., 0., 0.]
                
                if color_patch == 1:
                    # color a single pixel
                    img[r, c] = labels[i] if labels is not None else [0., 0., 0.]
                else:   
                    # color a 3*3 patch
                    r = np.clip(r, 1, img.shape[0]-2)
                    c = np.clip(c, 1, img.shape[1]-2)
                    if labels is not None and labels[i].max() == 0:
                        continue
                    img[r-1:r+2, c-1:c+2] = labels[i] if labels is not None else [0., 0., 0.]

        return img
    
    def find_proj_idx(self):
        """
        Find the projection index for each frame's camera view.
        """
        self.proj_idx = []
        self.proj_points = []
        for cam_idx in range(self.images.shape[0]):
            pcd, pcd_colors, indices = aggr_point_cloud_from_data(
                colors = np.expand_dims(self.images[cam_idx], 0),
                depths = np.expand_dims(self.depths[cam_idx], 0),
                Ks = np.expand_dims(self.intrinsics[cam_idx], 0),
                poses = np.expand_dims(self.extrinsics[cam_idx], 0),
                downsample=False, out_o3d=False, keep_shape=True)
            self.proj_idx.append(indices)
            self.proj_points.append(pcd)

    def assign_labels(self, proj_idx, neighbor_idx, labels, bg_val=1.):
        """
        Assign labels to each pixel based on the nearest neighbor in a fused pointcloud.

        Args:
        - proj_idx (numpy.ndarray): An array of shape (H, W) with indices to the projected points.
        - neighbor_idx (numpy.ndarray): An array of shape (N,) with indices of the closest neighbor in the fused pointcloud.
        - labels (numpy.ndarray): An array of shape (M, *) with labels for each point in the fused pointcloud.

        Returns:
        - pixel_labels (numpy.ndarray): An array of shape (H, W, *) with labels for each pixel.
        """
        H, W = proj_idx.shape
        # Initialize the output array specified bg val
        pixel_labels = np.full((H, W) + labels.shape[1:], bg_val, dtype=labels.dtype)
        
        # Find all valid pixel indices in proj_idx (where proj_idx is not -1)
        valid_mask = proj_idx != -1
        valid_pixels = np.nonzero(valid_mask)
        
        # Get the indices in the neighbor_idx array corresponding to the valid projected points
        valid_proj_indices = proj_idx[valid_mask]
        valid_neighbor_indices = neighbor_idx[valid_proj_indices]
        
        # Assign the labels to the valid pixels
        pixel_labels[valid_pixels] = labels[valid_neighbor_indices]

        return pixel_labels
    
    def view_projection_to_cam_3d(self, frame_idx, labels=None, label_discrete=False, n_neighbors=1,bg_val=1):
        """
        Project labels to camera view by finding the closest point to the projected point.
        """
        frame_proj_points = self.proj_points[frame_idx]
        frame_proj_idx = self.proj_idx[frame_idx]

        tree = cKDTree(self.points)
        _, neighbor_idx = tree.query(frame_proj_points, k=n_neighbors)

        if labels is None:
            labels = self.colors
        
        assigned_labels = self.assign_labels(frame_proj_idx, neighbor_idx, labels, bg_val=bg_val)
        return assigned_labels
    
    def visualize_cluster(self, cluster_results, frame_idx=0, start_pallete=0, visualize_labels=None, return_color_names=False, proj_3d=False):
        """
        convert cluster results to color labels. Calls view_projection_to_cam() to visualize.
        args:
            - cluster_results: (self.points.shape[0],) array, cluster labels for each point
            - frame_idx: int, index of the frame to visualize
            - start_pallete: int, start index of the color pallete to use (used to skip first colors)
            - return_color_names: bool, if True, returns the color names of the clusters.
        """
        
        # get color palette
        palette = get_palette()[start_pallete:]
        used_colors = {}

        if visualize_labels is None:
            visualize_labels = np.unique(cluster_results)

        cluster_colors = np.zeros((cluster_results.shape[0], 3), dtype=np.int16)
        unique_labels = list(np.unique(cluster_results))
        if -1 in unique_labels:
            unique_labels.remove(-1)
            unique_labels.append(-1) # put -1 at the end
        for i, label in enumerate(unique_labels):
            if label == -1 or label not in visualize_labels:
                cluster_colors[cluster_results == label] = [0, 0, 0]
                continue
            cluster_colors[cluster_results == label] = palette[i][0]
            used_colors[label] = palette[i][1]
        # imgs = [self.view_projection_to_cam(i, labels=cluster_colors) for i in range(self.images.shape[0])]
        cluster_colors = cluster_colors / 255.

        if proj_3d:
            img = self.view_projection_to_cam_3d(frame_idx, labels=cluster_colors)
        else:
            img = self.view_projection_to_cam(frame_idx, labels=cluster_colors, overlay_orig=0.5)

        if not return_color_names:
            return img
        else:
            return img, used_colors
        
    def cosine_similarity(self, features, query):
        """
        Calculates cosine similarity between a query vector and each vector in a feature matrix.

        Args:
        - features (numpy.ndarray): Matrix of shape (M, 1024).
        - query (numpy.ndarray): Vector of shape (1024,).

        Returns:
        - numpy.ndarray: Cosine similarities of shape (M,).
        """
        # Normalize query and compute dot product with features
        query_norm = query / np.linalg.norm(query)
        dot_products = np.dot(features, query_norm)
        
        # Normalize features and compute cosine similarities
        feature_norms = np.linalg.norm(features, axis=1)

        # turn 0 values to 1 to avoid division by zero
        feature_norms[feature_norms == 0] = 1
        similarities = dot_products / feature_norms

        return similarities
        
    def aggregate_cluster_feature(self, cluster_results, method='mean', return_similarities=False):
        """
        Aggregate the feature for given cluster.
        cluster_results: (N,) array, cluster labels for each point
        """
        if self.fused_weighted_features is None:
            raise ValueError("fused_weighted_features is None. Run compute_fused_weighted_features() first.")
        
        unique_labels = list(np.unique(cluster_results))
        if -1 in unique_labels: unique_labels.remove(-1)

        cluster_features = {}
        similarities = {}
        for label in unique_labels:
            mask = cluster_results == label
            # now assume using mean as aggregation method
            cluster_features[label] = np.mean(self.fused_weighted_features[mask], axis=0) # (feat_dim,)

            if return_similarities:
                similarities[label] = self.cosine_similarity(self.fused_weighted_features, cluster_features[label])
        
        if return_similarities:
            return cluster_features, similarities
        return cluster_features


###########################################
## Helper functions -- Process data
###########################################

def load_data_from_h5(instance, dinov2, use_data_link_segs=False):
    """
    Load the data dict from an H5 group instance.
    Also obtains DINO features.
    
    Args:
        instance: h5py.Group object containing the instance data
        dinov2: DINOv2 model for feature extraction
    """
    # Load RGB images
    colors = instance['rgb'][:]  # Shape: (N, 378, 378, 3)
    
    # Load depth maps and convert from millimeters to meters
    depths = instance['depth'][:].astype(np.float32) / 1000.0 # Shape: (N, 378, 378)
    # print("max depth after processing: ", depths.max())
    
    # Load camera parameters
    T = np.array([[1, 0, 0, 0], 
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]])
    
    extrinsics = instance['extrinsics'][:]  # Shape: (N, 4, 4)
    for i in range(extrinsics.shape[0]):
        extrinsics[i] = T @ extrinsics[i]
    intrinsics = instance['intrinsics'][:]  # Shape: (N, 3, 3)
    
    if use_data_link_segs is False:
        # Create link segmentation masks (foreground = 1, background = -1) for objects with no link annotations
        # Define foreground as any pixel that is not exactly white (255, 255, 255)
        white_mask = np.all(colors == 255, axis=-1)
        link_segs = np.logical_not(white_mask).astype(np.uint8)
        link_segs = link_segs * 2 - 1 
    else:
        print("using actual data link segs")
        if 'link_segs' not in instance:
            raise ValueError("link_segs not found in instance, please set use_data_link_segs to False")
        link_segs = instance['link_segs'][:]

    # get DINO features batch by batch
    batch_size = 4 # adjust this if you have OOM
    features = []
    for i in range(0, colors.shape[0], batch_size):
        end_idx = min(i + batch_size, colors.shape[0])
        features.append(get_dino_features(dinov2, colors[i:end_idx], repeat_to_orig_size=True))
        print(f'Processed {end_idx} images')
    features = np.concatenate(features, axis=0)
    
    return {
        'colors': colors,           # (N, H, W, 3) RGB images
        'depths': depths,           # (N, H, W) depth maps in meters
        'link_segs': link_segs,     # (N, H, W) binary segmentation masks
        'intrinsics': intrinsics,   # (N, 3, 3) camera intrinsic matrices
        'extrinsics': extrinsics,    # (N, 4, 4) camera extrinsic matrices
        'features': features        # (N, H, W, 1024) DINO features
    }    

def create_fusion(instance, dinov2, normalize_extent=None, use_data_link_segs=False):
    """
    Create a fusion object from the data in an H5 group instance.
    
    Args:
        instance: h5py.Group object containing the instance data
        dinov2: DINOv2 model for feature extraction
        normalize_extent: extent to normalize the point cloud to
    
    Returns:
        fusion: Fusion object. No frame fused.
    """
    data = load_data_from_h5(instance, dinov2, use_data_link_segs)
    pcd = aggr_point_cloud_from_data(data['colors'], data['depths'], 
                                    data['intrinsics'], data['extrinsics'], 
                                    downsample=True, radius=0.001, n_points=50000)
    
    # if extent is passed in, rescale the pcd to the extent
    if normalize_extent is not None:
        if isinstance(normalize_extent, (int, float)):
            normalize_extent = np.array([normalize_extent] * 3)
        else:
            if len(normalize_extent) != 3:
                raise ValueError('rescale_extent must be a scalar or a 3-element list')
        normalized_pcd, rescale_params = normalize_pointcloud(pcd, normalize_extent)
    else:
        normalized_pcd = pcd
        rescale_params = None
    
    fusion = Fusion(normalized_pcd, data, rescale_params=rescale_params)
    return fusion
