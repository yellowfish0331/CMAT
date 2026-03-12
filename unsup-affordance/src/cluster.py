"""
Script to create and visualize clustering results.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from kmeans_pytorch import kmeans


def get_cluster_feat(fusion, pca_dim=3, use_loc=0.0, select_mask=None):
    """
    Get the feature for clustering for a fusion object.
    Input:
        fusion: a Fusion object; make sure have features fused already
        pca_dim: the dimension of PCA
        select_mask: the index of the features to select. T/F array of shape (N,)
    """
    if fusion.fused_weighted_features is None:
        raise ValueError("Features not fused yet")

    if pca_dim > 0:
        pca = PCA(n_components=pca_dim)
        pca.fit(fusion.fused_weighted_features[select_mask])
        features_pca = pca.transform(fusion.fused_weighted_features[select_mask])
        features_pca = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min())
    else: # no PCA, use raw features with normalization
        features_pca = fusion.fused_weighted_features[select_mask]
        features_pca = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min())

    # prepare voxel-space distance
    if use_loc > 0.:
        feature_voxel = fusion.points[select_mask]
        feature_voxel = (feature_voxel - feature_voxel.min()) / (feature_voxel.max() - feature_voxel.min())
        X = np.concatenate((1.0 * features_pca, use_loc * feature_voxel), axis=1)
    else:
        X = features_pca

    feat = torch.from_numpy(X)
    return feat

def k_means_cluster(fusion, num_clusters=4, pca_dim=3, use_loc=None, frame_idx=0, return_color_names=True, proj_3d= False):
    """
    Cluster the features of a fusion object, visualize the results.
    """
    # prepare feature
    feat = get_cluster_feat(fusion, pca_dim=pca_dim, use_loc=use_loc)

    # run cluster
    cluster_ids, centers = kmeans(X=feat, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
    print(f"Kmeans with {num_clusters} clusters done.")
    cluster_results = cluster_ids.detach().numpy()

    # visualize
    ret = fusion.visualize_cluster(cluster_results, frame_idx=frame_idx, return_color_names=return_color_names, proj_3d=proj_3d)
    
    if return_color_names:
        img, used_colors = ret
        return cluster_results, img, used_colors
    else:
        img = ret
        return cluster_results, img

def run_KMeans(feat, num_clusters=4, iter_limit=200, seed=42):
    """
    Run KMeans on the feature.
    params: 
        feat: tensor of shape (N, D)
    return: 
        cluster_results: numpy array of shape (N,)
    """
    # cluster_ids, centers = kmeans(X=feat, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'), iter_limit=iter_limit, seed=seed)
    cluster_ids, centers = kmeans(X=feat, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
    print(f"Kmeans with {num_clusters} clusters done.")
    cluster_results = cluster_ids.detach().numpy().astype(np.int8)

    return cluster_results

def run_MeanShift(feat, bandwidth=0.2):
    """
    Run MeanShift on the feature.
    params: 
        feat: tensor of shape (N, D)
    return: 
        cluster_results: numpy array of shape (N,)
    """
    mean_shift = MeanShift(bandwidth=bandwidth, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=16, max_iter=100)
    mean_shift.fit(feat)
    cluster_results = mean_shift.labels_.astype(np.int8)

    return cluster_results

def cluster(fusion, pca_dim=3, use_loc=None, 
            frame_idx=0, return_color_names=True, proj_3d=False, 
            min_num_clusters=None, enable_per_link_cluster=False):
    """
    Run clustering on fusion fused features.
    params:
        pca_dim: int, the dimension of PCA
        use_loc: float, the weight of xyz
        frame_idx: int, the frame index to visualize result
        return_color_names: bool, whether to return color names
        proj_3d: bool, whether to project to 3D during cluster visualization
        min_num_clusters: int, the minimum number of clusters
        enable_per_link_cluster: bool, whether to run per-link clustering for articulated obj
    """
    def run_cluster(cluster_feat, min_num_clusters=None, max_num_clusters=20):
        cluster_feat_np = cluster_feat.cpu().numpy()
        # first run meanshift
        cluster_results = run_MeanShift(cluster_feat_np)

        if min_num_clusters is None or (len(np.unique(cluster_results)) >= min_num_clusters and len(np.unique(cluster_results)) <= max_num_clusters):
            return cluster_results
        elif len(np.unique(cluster_results)) < min_num_clusters: # too few clusters
            return run_KMeans(cluster_feat, num_clusters=min_num_clusters)
        else: # too many clusters
            return run_KMeans(cluster_feat, num_clusters=max_num_clusters)

    if enable_per_link_cluster is True:
        link_seg_label = fusion.fused_link_segs
    else: # dummy label for global clustering
        link_seg_label = np.ones(fusion.fused_weighted_features.shape[0], dtype=np.int8)

    unique_link_seg_label = list(np.unique(link_seg_label))
    if -1 in unique_link_seg_label: # remove -1 (for invalid points)
        unique_link_seg_label.remove(-1)
    if min_num_clusters is not None: # assume evenly split threshold between links
        min_num_clusters = int(np.ceil(min_num_clusters / len(unique_link_seg_label)))
    max_num_clusters = int(np.floor(20 / len(unique_link_seg_label)))

    cluster_results = np.full(fusion.points.shape[0], -1, dtype=np.int8)
    current_max_cluster_id = 0 

    for link_label in unique_link_seg_label:
        select_mask = (link_seg_label == link_label) # select by link label
        select_idx = np.arange(len(select_mask))[select_mask] # actual index of selected items

        # get cluster feature
        cluster_feat = get_cluster_feat(fusion, pca_dim=pca_dim, use_loc=use_loc, select_mask=select_mask)
        # run cluster for this link
        cur_cluster_results = run_cluster(cluster_feat, min_num_clusters=min_num_clusters, max_num_clusters=max_num_clusters)
        cur_cluster_results = cur_cluster_results + current_max_cluster_id

        # update cluster results
        cluster_results[select_idx] = cur_cluster_results
        current_max_cluster_id = np.max(cluster_results) + 1

    # visualize
    ret = fusion.visualize_cluster(cluster_results, frame_idx=frame_idx, return_color_names=return_color_names, proj_3d=proj_3d)

    if return_color_names:
        img, used_colors = ret
        return cluster_results, img, used_colors
    else:
        img = ret
        return cluster_results, img