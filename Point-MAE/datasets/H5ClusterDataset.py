import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .build import DATASETS


@DATASETS.register_module()
class H5Cluster(Dataset):
    """
    Dataset for loading point clouds and per-point fused features from H5 files.

    Expects each H5 file to contain per-instance groups with at least:
        - 'cluster_points': [N, 3] float32 point coordinates
        - 'cluster_labels': [N] int32 cluster labels
    Optionally can contain:
        - 'cluster_features': [N, D] float32 fused 2D features

    Args in cfg.others:
        h5_root (str): root directory containing category folders or h5 folder
        categories (list[str] | None): list of category names; if None, scan all .h5
        subset (str): 'train' | 'test' (used only for shuffling/drop_last outside)
        npoints (int): target number of points to sample per instance
        use_features (bool): whether to load per-point features if available
        feature_key (str): dataset key for features, default 'cluster_features'
        label_key (str): dataset key for labels, default 'cluster_labels'
        points_key (str): dataset key for points, default 'cluster_points'
    """

    def __init__(self, config):
        """
        Initialize dataset from a config EasyDict.
        """
        super().__init__()
        self.h5_root = config.h5_root
        self.subset = getattr(config, 'subset', 'train')
        self.npoints = int(getattr(config, 'npoints', 1024))
        self.categories = getattr(config, 'categories', None)
        self.use_features = bool(getattr(config, 'use_features', True))
        self.feature_key = getattr(config, 'feature_key', 'cluster_features')
        self.label_key = getattr(config, 'label_key', 'cluster_labels')
        self.points_key = getattr(config, 'points_key', 'cluster_points')

        self.samples = []
        self._index_h5()

    def _index_h5(self):
        if self.categories is None:
            h5_files = glob.glob(os.path.join(self.h5_root, '*.h5'))
        else:
            h5_files = []
            for cat in self.categories:
                fp = os.path.join(self.h5_root, f'{cat}.h5')
                if os.path.exists(fp):
                    h5_files.append(fp)
        for fp in sorted(h5_files):
            try:
                with h5py.File(fp, 'r') as f:
                    for inst_key in f.keys():
                        grp = f[inst_key]
                        if self.points_key not in grp or self.label_key not in grp:
                            continue
                        if self.use_features and (self.feature_key not in grp):
                            continue
                        has_feat = (self.feature_key in grp)
                        self.samples.append((fp, inst_key, has_feat))
            except Exception:
                continue
        if not self.samples:
            raise RuntimeError(f'No valid H5Cluster samples found under {self.h5_root}')

    def __len__(self):
        return len(self.samples)

    def _fps(self, pts, npoints):
        num = pts.shape[0]
        if num <= npoints:
            idx = np.arange(num, dtype=np.int64)
            if num < npoints:
                pad = np.random.choice(num, npoints - num, replace=True)
                idx = np.concatenate([idx, pad], axis=0)
            return idx
        idx = np.zeros(npoints, dtype=np.int64)
        distances = np.ones(num, dtype=np.float32) * 1e10
        farthest = np.random.randint(0, num)
        for i in range(npoints):
            idx[i] = farthest
            centroid = pts[farthest]
            dist = np.sum((pts - centroid) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = np.argmax(distances)
        return idx

    def __getitem__(self, index):
        h5_path, inst_key, has_feat = self.samples[index]
        with h5py.File(h5_path, 'r') as f:
            grp = f[inst_key]
            pts = grp[self.points_key][()].astype(np.float32)
            labels = grp[self.label_key][()].astype(np.int64)
            feats = None
            if has_feat:
                feats = grp[self.feature_key][()].astype(np.float32)

        assert pts.ndim == 2 and pts.shape[1] == 3
        assert labels.ndim == 1 and labels.shape[0] == pts.shape[0]
        if feats is not None:
            assert feats.shape[0] == pts.shape[0]

        sel = self._fps(pts, self.npoints)
        pts_sel = pts[sel]
        labels_sel = labels[sel]
        feats_sel = feats[sel] if feats is not None else None

        taxonomy_id = 'H5Cluster'
        model_id = os.path.basename(h5_path) + '/' + inst_key

        if self.use_features:
            # always return (points, feats, labels)
            if feats_sel is None:
                raise RuntimeError(f"Features not found for sample {model_id} while use_features=True")
            data = (torch.from_numpy(pts_sel), torch.from_numpy(feats_sel), torch.from_numpy(labels_sel))
            return taxonomy_id, model_id, data
        else:
            # return only points tensor (consistent with ShapeNet branch)
            return taxonomy_id, model_id, torch.from_numpy(pts_sel)


