import glob
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

from cluster import cluster
from fusion import create_fusion
from utils.file_utils import save_image, store_or_update_dataset
from utils.img_utils import load_pretrained_dino

sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fuse_all_frames(fusion):
    """Fuse all rendered views into the point cloud."""
    for frame_idx in range(fusion.num_frames):
        fusion.fuse_frame(frame_idx)


def process_instance(
    instance: h5py.Group,
    dinov2,
    proposal_path=None,
    use_data_link_segs=False,
    visualize_frame_idx=0,
    pca_dim=3,
    use_loc=0.0,
    min_num_clusters=5,
    enable_per_link_cluster=True,
    normalize_extent=None,
):
    """
    Create a fused 3D representation from HDF5, run clustering, and store only
    the stage-1 artifacts used by CMAT.
    """
    fusion = create_fusion(
        instance,
        dinov2,
        normalize_extent=normalize_extent,
        use_data_link_segs=use_data_link_segs,
    )
    fuse_all_frames(fusion)

    frame_idx = int(np.clip(visualize_frame_idx, 0, fusion.num_frames - 1))
    cluster_results, proposal_img, used_colors = cluster(
        fusion,
        pca_dim=pca_dim,
        use_loc=use_loc,
        frame_idx=frame_idx,
        return_color_names=True,
        proj_3d=True,
        min_num_clusters=min_num_clusters,
        enable_per_link_cluster=enable_per_link_cluster,
    )

    cluster_features, similarities = fusion.aggregate_cluster_feature(
        cluster_results,
        return_similarities=True,
    )

    unique_labels = [label for label in np.unique(cluster_results) if label != -1]
    color_label_names = [used_colors[label] for label in unique_labels]
    feature_array = np.array([cluster_features[label] for label in unique_labels], dtype=np.float32)
    similarity_array = np.array([similarities[label] for label in unique_labels], dtype=np.float32)

    store_or_update_dataset(instance, "fused_points", fusion.points.astype(np.float32))
    store_or_update_dataset(instance, "fused_point_colors", fusion.colors.astype(np.float32))
    store_or_update_dataset(instance, "fused_features", fusion.fused_weighted_features.astype(np.float32))
    store_or_update_dataset(instance, "fused_feature_weights", fusion.fused_weights.astype(np.float32))
    store_or_update_dataset(instance, "cluster_labels", cluster_results.astype(np.int32))
    store_or_update_dataset(instance, "cluster_color_names", color_label_names)
    store_or_update_dataset(instance, "cluster_feature_means", feature_array)
    store_or_update_dataset(instance, "cluster_point_similarities", similarity_array, compression="gzip")

    if proposal_path is not None:
        save_image(proposal_img, proposal_path)


def process_category(
    category_h5_path,
    dinov2,
    proposal_save_dir,
    use_data_link_segs=False,
    visualize_frame_idx=0,
    pca_dim=3,
    use_loc=0.0,
    min_num_clusters=5,
    enable_per_link_cluster=True,
    normalize_extent=None,
):
    """Process every instance in one category H5 file."""
    category_name = os.path.basename(category_h5_path).split(".")[0]
    with h5py.File(category_h5_path, "r+") as h5_file:
        for instance_key in tqdm(list(h5_file.keys())):
            instance = h5_file[instance_key]
            proposal_path = os.path.join(proposal_save_dir, f"{category_name}_{instance_key}_cluster.png")
            process_instance(
                instance,
                dinov2,
                proposal_path=proposal_path,
                use_data_link_segs=use_data_link_segs,
                visualize_frame_idx=visualize_frame_idx,
                pca_dim=pca_dim,
                use_loc=use_loc,
                min_num_clusters=min_num_clusters,
                enable_per_link_cluster=enable_per_link_cluster,
                normalize_extent=normalize_extent,
            )
            h5_file.flush()


def main(args):
    """Run stage-1 CMAT data generation on one or more category H5 files."""
    base_dir = args.base_dir
    h5_dir = os.path.join(base_dir, "h5")
    if not os.path.exists(h5_dir):
        raise FileNotFoundError(f"h5 folder not found in {base_dir}, please pass in the parent of the h5 folder")

    if args.category_names:
        category_h5_paths = [os.path.join(h5_dir, f"{name}.h5") for name in args.category_names]
        for category_h5_path in category_h5_paths:
            if not os.path.exists(category_h5_path):
                raise ValueError(f"Category file not found: {category_h5_path}")
    else:
        category_h5_paths = glob.glob(os.path.join(h5_dir, "*.h5"))

    proposal_save_dir = os.path.join(base_dir, "cluster_proposals")
    os.makedirs(proposal_save_dir, exist_ok=True)

    normalize_extent = None
    if args.normalize_extent is not None:
        normalize_extent = args.normalize_extent

    dinov2 = load_pretrained_dino("dinov2_vits14", use_registers=True, torch_path=args.torch_path)

    for category_h5_path in category_h5_paths:
        print(f"Processing {category_h5_path}")
        process_category(
            category_h5_path,
            dinov2,
            proposal_save_dir=proposal_save_dir,
            use_data_link_segs=args.use_data_link_segs,
            visualize_frame_idx=args.visualize_frame_idx,
            pca_dim=args.pca_dim,
            use_loc=args.use_loc,
            min_num_clusters=args.min_num_clusters,
            enable_per_link_cluster=not args.disable_per_link_cluster,
            normalize_extent=normalize_extent,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--use_data_link_segs", action="store_true")
    parser.add_argument("--visualize_frame_idx", type=int, default=0)
    parser.add_argument("--pca_dim", type=int, default=3)
    parser.add_argument("--use_loc", type=float, default=0.0)
    parser.add_argument("--min_num_clusters", type=int, default=5)
    parser.add_argument("--disable_per_link_cluster", action="store_true")
    parser.add_argument(
        "--category_names",
        "-c",
        nargs="+",
        dest="category_names",
        default=None,
        help="Optional: one or more category names to process. If omitted, all categories in <base_dir>/h5 are processed.",
    )
    parser.add_argument("--normalize_extent", type=float, nargs="*", default=None)
    parser.add_argument("--torch_path", type=str, default=None, help="Path to torch model cache directory")
    main(parser.parse_args())
