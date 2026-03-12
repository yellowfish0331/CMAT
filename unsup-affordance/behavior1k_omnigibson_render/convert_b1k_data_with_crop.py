import os
import json
import h5py
import numpy as np
import cv2
from glob import glob
from pathlib import Path

def find_object_bounding_box(rgb_img, white_value=255):
    """
    Find the bounding box of non-white pixels in an RGB image.
    
    Args:
        rgb_img: RGB image as numpy array with shape (H, W, 3)
        white_value: Value to consider as white (default 255)
        
    Returns:
        tuple: (min_row, min_col, max_row, max_col) of the bounding box
    """
    # Find non-white pixels (pixels that aren't exactly [255, 255, 255])
    white_mask = np.all(rgb_img == white_value, axis=2)
    non_white = ~white_mask
    
    # Get indices of non-white pixels
    if not np.any(non_white):
        # If no non-white pixels, return full image bounds
        return 0, 0, rgb_img.shape[0], rgb_img.shape[1]
    
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    
    # Get min/max row/col indices
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    
    return min_row, min_col, max_row, max_col

def make_square_bbox(min_row, min_col, max_row, max_col, img_height, img_width):
    """
    Make a bounding box square by expanding it.
    
    Args:
        min_row, min_col, max_row, max_col: Bounding box coordinates
        img_height, img_width: Original image dimensions
        
    Returns:
        tuple: (min_row, min_col, max_row, max_col) of the square bounding box
    """
    # Calculate center of the bounding box
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    
    # Get the current width and height
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    # Take the larger dimension
    size = max(height, width)
    
    # Add some padding (10%)
    size = int(size * 1.1)
    
    # Calculate new boundaries
    half_size = size // 2
    new_min_row = max(0, center_row - half_size)
    new_min_col = max(0, center_col - half_size)
    new_max_row = min(img_height - 1, center_row + half_size)
    new_max_col = min(img_width - 1, center_col + half_size)
    
    # If we hit image boundaries, adjust to maintain square shape
    if new_min_row == 0:
        new_max_row = min(img_height - 1, new_min_row + size - 1)
    if new_min_col == 0:
        new_max_col = min(img_width - 1, new_min_col + size - 1)
    if new_max_row == img_height - 1:
        new_min_row = max(0, new_max_row - size + 1)
    if new_max_col == img_width - 1:
        new_min_col = max(0, new_max_col - size + 1)
    
    # Final size check and adjustment to ensure squareness
    actual_height = new_max_row - new_min_row + 1
    actual_width = new_max_col - new_min_col + 1
    if actual_height > actual_width:
        diff = actual_height - actual_width
        new_min_col = max(0, new_min_col - diff // 2)
        new_max_col = min(img_width - 1, new_max_col + (diff - diff // 2))
    elif actual_width > actual_height:
        diff = actual_width - actual_height
        new_min_row = max(0, new_min_row - diff // 2)
        new_max_row = min(img_height - 1, new_max_row + (diff - diff // 2))
    
    return new_min_row, new_min_col, new_max_row, new_max_col

def crop_and_scale_image(rgb_img, depth_img, seg_img, intrinsics, target_size=364):
    """
    Crop the image to the smallest square containing all non-white pixels,
    scale to target size, and adjust intrinsics accordingly.
    
    Args:
        rgb_img: RGB image as numpy array with shape (H, W, 3)
        depth_img: Depth image as numpy array with shape (H, W)
        seg_img: Segmentation image as numpy array with shape (H, W)
        intrinsics: 3x3 camera intrinsics matrix
        target_size: Target size for the output image (square)
        
    Returns:
        tuple: (cropped_rgb, cropped_depth, cropped_seg, new_intrinsics, crop_params)
              where crop_params is (min_row, min_col, max_row, max_col, scale_factor)
    """
    # Find bounding box of non-white pixels
    min_row, min_col, max_row, max_col = find_object_bounding_box(rgb_img)
    
    # Make bounding box square
    min_row, min_col, max_row, max_col = make_square_bbox(
        min_row, min_col, max_row, max_col, rgb_img.shape[0], rgb_img.shape[1])
    
    # Crop RGB, depth, and segmentation images
    cropped_rgb = rgb_img[min_row:max_row+1, min_col:max_col+1].astype(np.uint16)
    cropped_depth = depth_img[min_row:max_row+1, min_col:max_col+1].astype(np.float32)
    cropped_seg = seg_img[min_row:max_row+1, min_col:max_col+1].astype(np.uint16)
    
    # Get current size and calculate scale factor
    current_size = cropped_rgb.shape[0]  # Should be square now
    scale_factor = target_size / current_size
    
    # Resize RGB image
    cropped_rgb = cv2.resize(cropped_rgb, (target_size, target_size), 
                             interpolation=cv2.INTER_LINEAR)
    
    # Resize depth image (nearest neighbor to avoid interpolation artifacts)
    cropped_depth = cv2.resize(cropped_depth, (target_size, target_size), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Resize segmentation image (nearest neighbor to preserve labels)
    cropped_seg = cv2.resize(cropped_seg, (target_size, target_size), 
                             interpolation=cv2.INTER_NEAREST)
    
    # Adjust camera intrinsics
    new_intrinsics = intrinsics.copy()
    
    # 1. Adjust for crop (shift principal point)
    new_intrinsics[0, 2] = intrinsics[0, 2] - min_col  # cx
    new_intrinsics[1, 2] = intrinsics[1, 2] - min_row  # cy
    
    # 2. Adjust for scale
    new_intrinsics[0, 0] *= scale_factor  # fx
    new_intrinsics[1, 1] *= scale_factor  # fy
    new_intrinsics[0, 2] *= scale_factor  # cx
    new_intrinsics[1, 2] *= scale_factor  # cy
    new_intrinsics = new_intrinsics.astype(np.float32)
    
    # Return cropped images, new intrinsics, and parameters for reference
    crop_params = (min_row, min_col, max_row, max_col, scale_factor)
    return cropped_rgb, cropped_depth, cropped_seg, new_intrinsics, crop_params

def convert_data(source_dir, target_file, target_size=512):
    """
    Convert data from multiple scan.h5 files and their associated metadata.json
    into a single h5 file with the specified format.
    
    Args:
        source_dir: Path to the 'coffee_cup' directory
        target_file: Path where the output h5 file will be saved
        target_size: Target size for cropped and resized images (default 512)
    """
    print(f"Processing data from {source_dir}")
    print(f"Output will be saved to {target_file}")
    
    # Get all instance directories
    instance_dirs = []
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):  # 6-letter folder names
            scan_file = os.path.join(item_path, "scan.h5")
            meta_file = os.path.join(item_path, "metadata.json")
            if os.path.exists(scan_file) and os.path.exists(meta_file):
                instance_dirs.append(item_path)
    
    print(f"Found {len(instance_dirs)} instance directories")
    
    # Create the output h5 file
    with h5py.File(target_file, 'w') as target_h5:
        # Process each instance
        for idx, instance_dir in enumerate(instance_dirs):
            instance_name = f"instance_{idx}"
            print(f"Processing {os.path.basename(instance_dir)} -> {instance_name}")
            
            # Create instance group
            instance_group = target_h5.create_group(instance_name)
            
            # Read metadata.json for intrinsics
            meta_file = os.path.join(instance_dir, "metadata.json")
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Get camera intrinsics
            cam_intrinsic = np.array(metadata['scan']['cam_intrinsic'])
            
            # Read scan.h5 data
            scan_file = os.path.join(instance_dir, "scan.h5")
            with h5py.File(scan_file, 'r') as scan_h5:
                # Get the data, downsampling by 2 in the frame dimension
                rgb_data = scan_h5['rgb'][()][::2]
                depth_data = scan_h5['depth'][()][::2]
                link_segs_data = scan_h5['link_seg'][()][::2]
                extrinsics_data = scan_h5['cam_extrinsic'][()][::2].astype(np.float32)
                
                # Convert depth to mm and set to uint16
                depth_data = (depth_data * 1000).astype(np.uint16)
                
                # Create datasets for processed data
                num_frames = rgb_data.shape[0]
                
                # Create arrays to store cropped and resized data
                cropped_rgb = np.zeros((num_frames, target_size, target_size, 3), dtype=np.uint8)
                cropped_depth = np.zeros((num_frames, target_size, target_size), dtype=np.uint16)
                cropped_segs = np.zeros((num_frames, target_size, target_size), dtype=link_segs_data.dtype)
                new_intrinsics = np.zeros((num_frames, 3, 3), dtype=np.float32)
                crop_params_list = []
                
                # Process each frame
                for frame_idx in range(num_frames):
                    # Get current frame data
                    rgb_frame = rgb_data[frame_idx]
                    depth_frame = depth_data[frame_idx]
                    seg_frame = link_segs_data[frame_idx]
                    
                    # Crop and resize
                    cropped_rgb_frame, cropped_depth_frame, cropped_seg_frame, frame_intrinsics, crop_params = crop_and_scale_image(
                        rgb_frame, depth_frame, seg_frame, cam_intrinsic, target_size)
                    
                    # Store processed data
                    cropped_rgb[frame_idx] = cropped_rgb_frame
                    cropped_depth[frame_idx] = cropped_depth_frame
                    cropped_segs[frame_idx] = cropped_seg_frame
                    new_intrinsics[frame_idx] = frame_intrinsics
                    crop_params_list.append(crop_params)
                
                # Save processed data to the output file
                instance_group.create_dataset('rgb', data=cropped_rgb)
                instance_group.create_dataset('depth', data=cropped_depth)
                instance_group.create_dataset('link_segs', data=cropped_segs)
                instance_group.create_dataset('extrinsics', data=extrinsics_data)
                instance_group.create_dataset('intrinsics', data=new_intrinsics)
                
                # Save crop parameters as a separate dataset for reference
                crop_params_array = np.array(crop_params_list, dtype=np.float32)
                instance_group.create_dataset('crop_params', data=crop_params_array)
            
            # Add metadata
            instance_group.attrs['category'] = metadata['object']['category']
            instance_group.attrs['model'] = metadata['object']['model']
    
    print(f"Conversion completed. Output saved to {target_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--target_size", type=int, default=378)
    args = parser.parse_args()
    
    # Create h5 output directory
    h5_output_dir = os.path.join(args.data_root, "h5")
    os.makedirs(h5_output_dir, exist_ok=True)
    
    # Get all category directories
    category_dirs = []
    for item in os.listdir(args.data_root + "/og_raw"):
        item_path = os.path.join(args.data_root + "/og_raw", item)
        if os.path.isdir(item_path) and item != "h5":  # Skip the h5 output directory
            category_dirs.append(item_path)
    
    print(f"Found {len(category_dirs)} category directories")
    
    # Process each category
    for category_dir in category_dirs:
        category_name = os.path.basename(category_dir)
        output_file = os.path.join(h5_output_dir, f"{category_name}.h5")
        
        print(f"\n=== Processing category: {category_name} ===")
        
        # Run the conversion for this category
        convert_data(category_dir, output_file, args.target_size)
        
        print(f"Completed processing {category_name}")
    
    print(f"\nAll categories processed. Output files saved in: {h5_output_dir}")