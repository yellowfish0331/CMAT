"""
Util functions related to pointclouds.
"""
import numpy as np
import open3d as o3d
import copy


####################################################
## Frame aggregation code
####################################################
def aggr_point_cloud_from_data(colors, depths, Ks, poses, downsample=True, radius=0.01, 
                               n_points=None, masks=None, out_o3d=True, keep_shape=False):
    """
    Aggregates a pointcloud from a set of rgbd images.
    Input:
        colors: [N, H, W, 3] numpy array in uint8
        depths: [N, H, W] numpy array in meters
        Ks: [N, 3, 3] numpy array
        poses: [N, 4, 4] numpy array
        masks: [N, H, W] numpy array in bool
        radius, n_points: downsampling parameters
        out_o3d: whether to output open3d or numpy pointcloud
    Output:
        aggr_pcd: downsampled aggregated pcd. open3d.geometry.PointCloud or [N, 3] numpy array
    """

    N, H, W, _ = colors.shape
    colors = colors / 255.
    start = 0
    end = N
    step = 1

    pcds = []
    pcd_colors = []
    for i in range(start, end, step):
        depth = depths[i]
        color = colors[i]
        K = Ks[i]
        cam_param = [K[0,0], K[1,1], K[0,2], K[1,2]] # fx, fy, cx, cy
        if masks is None:
            mask = (depth > 0) & (depth < 10)
            if not np.any(mask):  # avoid the case where mask is all FALSE
                print("Initial mask is all False, recalculating mask...")
                mask = (depth > 0) & (depth < depth.max())
        else:
            mask = masks[i] & (depth > 0)
        # mask = np.ones_like(depth, dtype=bool)
        if keep_shape:
            pcd, indices = depth2fgpcd(depth, mask, cam_param, keep_shape=keep_shape)
        else:
            pcd = depth2fgpcd(depth, mask, cam_param)
            indices = None
  
        pose = copy.deepcopy(poses[i])
        # pose = np.linalg.inv(pose)
        pose[:3, :3] = pose[:3, :3].T
        pose[:3, 3] = -pose[:3, :3] @ pose[:3, 3]
        # pose = pose @ T
        
        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T
        
        if out_o3d:
            pcd_o3d = np2o3d(trans_pcd, color[mask])
        else:
            pcd_np =  trans_pcd
            pcd_color = color[mask]
        
        # downsample
        if out_o3d:
            # if downsample:
            #     # radius = 0.01
            #     pcd_o3d = pcd_o3d.voxel_down_sample(radius)
            pcds.append(pcd_o3d)
        else:
            if downsample:
                pcd_np, pcd_color = voxel_downsample(pcd_np, radius, pcd_color)
            pcds.append(pcd_np)
            pcd_colors.append(pcd_color)

    if out_o3d:
        aggr_pcd = o3d.geometry.PointCloud()
        for pcd in pcds:
            aggr_pcd += pcd
        if downsample:
            # default is using voxel downsample
            if n_points is None or len(aggr_pcd.points) > n_points:
                aggr_pcd = aggr_pcd.voxel_down_sample(radius)
            if n_points is not None: # further downsample to n_points
                aggr_pcd = downsample_to_n_points(aggr_pcd, n_points)
        return aggr_pcd
    else: # return numpy array
        pcds = np.concatenate(pcds, axis=0)
        pcd_colors = np.concatenate(pcd_colors, axis=0)
        return pcds, pcd_colors, indices

def downsample_to_n_points(pcd, n_points):
    """
    Downsample a point cloud to n_points
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None

    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
        return np2o3d(points, colors)
    
    return pcd

def depth2fgpcd(depth, mask, cam_params, keep_shape=False):
    # depth: (h, w)
    # fgpcd: (n, 3) 
    # mask: (h, w)
    # indices: (h, w) idx of point in fgpcd corresponding to each pixel. -1 if no point
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]

    if keep_shape:
        # Create index array
        indices = -1 * np.ones((h, w), dtype=int)  # Start with -1 for all pixels
        # Find the 1D indices of foreground pixels
        foreground_indices = np.flatnonzero(mask)
        # Map 1D indices to their positions in the fgpcd array
        indices_flat = indices.flatten()
        indices_flat[foreground_indices] = np.arange(mask.sum())
        indices = indices_flat.reshape(h, w)
        return fgpcd, indices
    
    return fgpcd

def np2o3d(pcd, color=None):
    """
    Convert numpy array to open3d point cloud
    """
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

def voxel_downsample(pcd, voxel_size, pcd_color=None):
    """
    Wrapper function to voxel downsample a point cloud in numpy array format
    Input:
        pcd, pcd_color: [N,3] numpy array
        voxel_size: float
    Output:
        pcd_down, pcd_color_down: [M,3] numpy array
    """
    pcd = np2o3d(pcd, pcd_color)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if pcd_color is not None:
        return np.asarray(pcd_down.points), np.asarray(pcd_down.colors)
    else:
        return np.asarray(pcd_down.points)
    
# TODO: currently directly adapting from D3Fields. clean and reorganize this function
def project_points_coords(pts, Rt, K, rescale_params=None):
    """
    Project 3D points to 2D coordinates
    Input:
        pts:  [pn,3]
        Rt:   [rfn,3,4]
        K:    [rfn,3,3]
    :return:
        coords:         [rfn,pn,2]
        valid_mask:   [rfn,pn]
        depth:          [rfn,pn,1]
    """
    import torch

    if rescale_params is not None:
        if 'translation' not in rescale_params or 'scale' not in rescale_params:
            print("invalid rescale_params")
        else:
            translation = rescale_params['translation']
            scale = rescale_params['scale']
            if abs(scale) < 1e-4:
                raise ValueError('scale is too small')
            pts = pts / scale - torch.tensor(translation, device=pts.device, dtype=pts.dtype)

    pn = pts.shape[0]
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=pts.dtype)],1)
    srn = Rt.shape[0]
    KRt = K @ Rt # rfn,3,4
    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=pts.dtype)
    last_row[:,:,3] = 1.0
    H = torch.cat([KRt,last_row],1) # rfn,4,4
    pts_cam = H[:,None,:,:] @ hpts[None,:,:,None]
    pts_cam = pts_cam[:,:,:3,0]
    depth = pts_cam[:,:,2:]
    invalid_mask = torch.abs(depth)<1e-4
    depth[invalid_mask] = 1e-3
    pts_2d = pts_cam[:,:,:2]/depth
    return pts_2d, ~(invalid_mask[...,0]), depth

def get_pointcloud_center_and_scale(input):
    """
    Calculate the center and scale (extent) of a point cloud.

    Parameters:
    - pcd: An Open3D point cloud object or a numpy array of shape (N, 3) representing the point cloud.

    Returns:
    - center: A tuple (x, y, z) representing the center of the point cloud.
    - scale: A tuple (scale_x, scale_y, scale_z) representing the scale or extent of the point cloud along each axis.
    """
    if isinstance(input, o3d.geometry.PointCloud):
        # Ensure the point cloud is not empty
        if not input.has_points():
            raise ValueError("The point cloud has no points.")

        # Convert Open3D point cloud to numpy array
        points = np.asarray(input.points)
    elif isinstance(input, np.ndarray):
        if input.shape[1] != 3:
            raise ValueError("The input array must have shape (N, 3).")
        points = input

    # Calculate the center of the point cloud
    center = np.mean(points, axis=0)

    # Calculate the scale (extent) of the point cloud
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    scale = max_bound - min_bound

    return {"center": center, "scale": scale}


def normalize_pointcloud(input, target_extent):
    """
    Normalize a point cloud by centering and scaling it to fit within a specified extent.

    Input:
    - input: An Open3D point cloud object or np.array of points
    - target_extent: A tuple (target_x, target_y, target_z) specifying the maximum extent in each axis.

    Output:
    - normalized_pcd: The normalized Open3D point cloud object or np.array of points.
    - translation: The translation applied to center the point cloud.
    - scale_factor: The uniform scale factor applied to fit the point cloud within the target extent.
    """
    if isinstance(input, o3d.geometry.PointCloud):
        if not input.has_points():
            raise ValueError("The point cloud has no points.")
        points = np.asarray(input.points)
    elif isinstance(input, np.ndarray):
        assert input.shape[1] == 3
        points = input
    else:
        raise ValueError("Input must be an Open3D point cloud or np.array of points.")

    # Calculate the current center and extent
    # points = np.asarray(pcd.points)
    current_center = np.mean(points, axis=0)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    current_extent = max_bound - min_bound

    # Translate the point cloud to center it at (0, 0, 0)
    translation = -current_center
    translated_points = points + translation
    # pcd.points = o3d.utility.Vector3dVector(translated_points)

    # Calculate the scale factor to fit the point cloud within the target extent, maintaining aspect ratio
    scale_factors = np.divide(target_extent, current_extent)
    scale_factor = np.min(scale_factors) # Use the smallest scale factor to ensure fit within target_extent

    # Scale the point cloud
    scaled_points = translated_points * scale_factor

    if isinstance(input, o3d.geometry.PointCloud):
        normalized_pcd = o3d.geometry.PointCloud()
        normalized_pcd.points = o3d.utility.Vector3dVector(scaled_points)
        return normalized_pcd, {"translation": translation, "scale": scale_factor}
    else:
        return scaled_points, {"translation": translation, "scale": scale_factor}

####################################################
## Visualization
####################################################

def visualize_pcd_in_plotly(pcd):
    """
    Visualize an Open3D point cloud in Plotly. Run in Jupyter Notebook.
    Input:
        pcd: open3d.geometry.PointCloud
    """
    import plotly.graph_objects as go

    # Convert Open3D.PointCloud to numpy array
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.colors else None

    # Determine the ranges for x, y, and z
    x_range = np.ptp(points[:,0])  # Peak to peak (max - min) for x
    y_range = np.ptp(points[:,1])  # Peak to peak (max - min) for y
    z_range = np.ptp(points[:,2])  # Peak to peak (max - min) for z
    max_range = np.max([x_range, y_range, z_range])

    # Calculate the center of the point cloud for each axis
    x_center = np.mean(points[:,0])
    y_center = np.mean(points[:,1])
    z_center = np.mean(points[:,2])

    # Set the same range for each axis based on the max_range
    x_lim = [x_center - max_range / 2, x_center + max_range / 2]
    y_lim = [y_center - max_range / 2, y_center + max_range / 2]
    z_lim = [z_center - max_range / 2, z_center + max_range / 2]

    # Create a scatter plot in Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:,0], 
        y=points[:,1], 
        z=points[:,2],
        mode='markers',
        marker=dict(
            size=2,  # Change the size of markers here
            color=colors,  # Add colors if available
            opacity=0.8
        )
    )])

    # Update layout of the figure to have the same scale for all axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_lim, autorange=False),
            yaxis=dict(range=y_lim, autorange=False),
            zaxis=dict(range=z_lim, autorange=False),
            aspectmode='cube'  # This forces the aspect ratio to be equal
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Show the figure in a Jupyter notebook
    fig.show()

    # return fig