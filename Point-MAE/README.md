# Point-MAE for CMAT Stage 2

This directory is the **stage-2 point encoder pretraining module** in the CMAT pipeline. It is derived from the official [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) codebase, but this open-source version keeps only the path required by this repository:

- masked point reconstruction pretraining
- `H5Cluster` loading for stage-1 exports
- the CMAT structural loss driven by fused stage-1 features

All unrelated branches from upstream Point-MAE, such as classification, few-shot evaluation, segmentation, and visualization utilities, have been removed here.

## Role in the pipeline

The full project is organized as:

1. `unsup-affordance`: export fused point features, cluster assignments, and similarity-aware structure into H5 files.
2. `Point-MAE`: pretrain the point encoder with reconstruction loss plus cross-modal affinity alignment.
3. `LAS`: use the pretrained point encoder for downstream affordance segmentation.

In this stage, each training sample contains:

- `cluster_points`: sampled 3D points
- `cluster_features`: per-point fused features exported by stage 1
- `cluster_labels`: cluster ids kept for traceability and possible analysis

## CMAT objective

The retained objective is:

```text
total_loss = recon_loss + lambda_struct * struct_loss
```

- `recon_loss`: Chamfer Distance (`cdl1` or `cdl2`) between reconstructed masked patches and ground-truth point patches.
- `struct_loss`: MSE between two Gram matrices:
  - the Gram matrix of visible Point-MAE token embeddings
  - the Gram matrix of group-level features aggregated from stage-1 `cluster_features`

The CMAT-specific step is the aggregation from point-level fused features to Point-MAE groups, followed by structure matching in token space. This is the main modification beyond upstream Point-MAE and is how stage 1 is connected to stage 2 in this repository.

## Minimal code path kept here

The retained training path is:

1. `main.py`
2. `tools/runner_pretrain.py`
3. `datasets/H5ClusterDataset.py`
4. `models/Point_MAE.py`

This is the only supported stage-2 entry in the trimmed open-source version.

## Dataset format

See [DATASET.md](./DATASET.md) for the expected H5 layout.

Important files:

- `cfgs/dataset_configs/H5Cluster.yaml`
- `cfgs/pretrain.yaml`
- `cfgs/pretrain_dw.yaml`

Before training, set `h5_root` in `cfgs/dataset_configs/H5Cluster.yaml` to the H5 directory exported from `unsup-affordance`.

## Training

Install the dependencies required by the original Point-MAE implementation first, especially:

- Chamfer Distance extension
- PointNet++ ops
- GPU kNN

Then launch pretraining with:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> python main.py --config cfgs/pretrain.yaml --exp_name <run_name>
```

To initialize from an existing checkpoint:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> python main.py --config cfgs/pretrain.yaml --exp_name <run_name> --start_ckpts <path/to/checkpoint>
```

## Acknowledgements

This module is built on top of the official Point-MAE codebase by Pang et al. We also acknowledge the upstream dependencies used by that project, including [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

## Reference

```bibtex
@inproceedings{pang2022masked,
  title={Masked autoencoders for point cloud self-supervised learning},
  author={Pang, Yatian and Wang, Wenxiao and Tay, Francis EH and Liu, Wei and Tian, Yonghong and Yuan, Li},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part II},
  pages={604--621},
  year={2022},
  organization={Springer}
}
```
