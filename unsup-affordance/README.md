## UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation

#### [[Project Page]](https://unsup-affordance.github.io/) [[Paper]](https://unsup-affordance.github.io/) 

[Yihe Tang](https://tangyihe.com/)<sup>1</sup>, [Wenlong Huang](https://wenlong.page)<sup>1</sup>, [Yingke Wang](https://www.wykac.com/)<sup>1</sup>, [Chengshu Li](https://www.chengshuli.me/)<sup>1</sup>, [Roy Yuan](https://www.linkedin.com/in/ryuan19)<sup>1</sup>, [Ruohan Zhang](https://ai.stanford.edu/~zharu/)<sup>1</sup>, [Jiajun Wu](https://jiajunwu.com/)<sup>1</sup>, [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)<sup>1</sup>

<sup>1</sup>Stanford University

### Source and Role in This Repository

This folder is derived from the official [unsup-affordance](https://github.com/TangYihe/unsup-affordance) / UAD codebase by Tang et al.

In this repository, it is used only as the **stage-0 data generation module** for the CMAT pipeline.

Kept functionality:
- render Behavior1K / Objaverse assets and convert them to HDF5
- compute DINO features and fuse them into a 3D point cloud
- cluster fused 3D features
- compute point-to-cluster similarity scores from fused features

Removed from this trimmed version:
- CLIP-based best-view selection
- VLM region matching
- language embedding generation
- downstream affordance model training / inference

Its role in the full repository is:
- generate fused 3D point features from multi-view renderings
- produce cluster-level features
- produce point-to-cluster similarity signals used as the stage-1 affinity output for the later CMAT/LAS pipeline


### Environment Setup
- (Optioal) If you are using [Omnigibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html) for rendering [Behavior1K](https://behavior.stanford.edu/behavior-1k) assets, or Blender for rendering [Objaverse](https://github.com/allenai/objaverse-xl) assets, please follow their installation guide respectively. 
  - Note the rendering libraries may have version conflicts with the fusion / clustering code, consider using a separate env in that case. 
- Create your conda environment and install torch
  ```
  conda create -n uad python=3.9
  conda activate uad
  pip install torch torchvision torchaudio
  ```
- Install unsup-affordance in the same conda env
  ```
  git clone https://github.com/TangYihe/unsup-affordance.git
  cd unsup-affordance
  pip install -r requirements.txt
  pip install -e .
  ```

### Object Rendering Pipeline
We provide code to render Behavior-1K assets with Omnigibson, or Objaverse assets with Blender. 

#### B1K assets
The code is in behavior1k_omnigibson_render. 
- Unzip ```qa_merged.zip```
- Render assets:
  ```
  python render.py --orientation_root ORI_ROOT --og_dataset_root OG_DATASET_ROOT --category_model_list selected_object_models.json --save_path YOUR_DATA_DIR
  ```
  Note: ORI_ROOT is the folder of your unzipped ```qa_merged/```. OG_DATASET_ROOT is your Omnigibson objects path, shall be ```YOUR_OG_PATH/omnigibson/data/og_dataset/objects```.
- Convert the renderings to .h5 format:
  ```
  python convert_b1k_data_with_crop.py --data_root YOUR_DATA_DIR
  ```

#### Objaverse assets
The code is in objaverse_blender_render. 
- Download the objaverse assets, run
  ```
  python objaverse_download_script.py --data_root YOUR_DATA_DIR --n N
  ```
  - N is the number of assets you want to download from each category. By default 50.
  - In our case study, we have used a subset from the lvis categories. You can change the category used in the script. 
- Filter out assets with transparent (no valid depth) or too simple texture, run
   ```
   python texture_filter.py --data_root YOUR_DATA_DIR
   ```
- Render the assets with Blender 
   ```
   blender --background \
   --python blender_script.py -- \
   --data_root YOUR_DATA_DIR \
   --engine BLENDER_EEVEE_NEXT \
   --num_renders 8 \
   --only_northern_hemisphere
   ```
- Convert the renderings to .h5 format
  ```
  python h5_conversion.py --data_root=YOUR_DATA_DIR
  ```


### Dataset Curation Pipeline
Pipeline to perform DINOv2 feature 3D fusion, clustering, and point-to-cluster similarity computation.

```
python src/pipeline.py --base_dir YOUR_DATA_DIR
```

The script writes the following stage-1 artifacts back into each HDF5 instance:
- `fused_points`
- `fused_point_colors`
- `fused_features`
- `fused_feature_weights`
- `cluster_labels`
- `cluster_color_names`
- `cluster_feature_means`
- `cluster_point_similarities`

Useful arguments:
- `--use_data_link_segs`: pass in when using Behavior-1K data
- `--visualize_frame_idx IDX`: choose which rendered view to use for cluster visualization
- `--category_names CATEGORY1 CATEGORY2`: only process certain categories
- `--min_num_clusters K`: lower bound for the number of clusters
- `--use_loc W`: add xyz location to clustering features with weight `W`

## Acknowledgements

This trimmed branch is based on the official UAD / unsup-affordance implementation by Tang et al.

We thank the original authors for releasing:
- [unsup-affordance](https://github.com/TangYihe/unsup-affordance)
- the UAD project page and paper linked above

