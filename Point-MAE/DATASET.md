## H5Cluster Dataset

This open-source version of `Point-MAE` only keeps the dataset format used by **CMAT stage 1**.

The pretraining dataloader expects a directory of `.h5` files:

```text
stage1_h5_root/
├── category_a.h5
├── category_b.h5
└── ...
```

Each H5 file should contain instance groups. Every instance group must provide:

- `cluster_points`: `float32`, shape `[N, 3]`
- `cluster_labels`: `int32` or `int64`, shape `[N]`

Optional but required for CMAT structural alignment:

- `cluster_features`: `float32`, shape `[N, D]`

An example layout is:

```text
category_a.h5
├── instance_000000/
│   ├── cluster_points
│   ├── cluster_labels
│   └── cluster_features
├── instance_000001/
│   ├── cluster_points
│   ├── cluster_labels
│   └── cluster_features
└── ...
```

## Field semantics

- `cluster_points` are the 3D points consumed by Point-MAE.
- `cluster_features` are the fused per-point features exported from stage 1 (`unsup-affordance`).
- `cluster_labels` record the cluster assignment and are kept for traceability; the current CMAT loss does not directly consume them.

During loading, the dataset samples `npoints` points per instance and keeps points, labels, and features aligned.

## Config

The dataset config lives in:

- `cfgs/dataset_configs/H5Cluster.yaml`

Set:

- `h5_root`: path to your stage-1 H5 directory
- `npoints`: number of sampled points per instance
- `feature_key`, `label_key`, `points_key`: H5 field names if your export uses custom keys

## Pipeline connection

This dataset is the bridge between stage 1 and stage 2:

1. `unsup-affordance` exports fused point-level features into H5 files.
2. `H5ClusterDataset` loads those files.
3. `runner_pretrain.py` forwards `cluster_features` into `Point_MAE`.
4. `Point_MAE` aggregates point-level fused features onto Point-MAE groups and computes reconstruction loss plus the CMAT structural loss.
