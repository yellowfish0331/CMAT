#!/usr/bin/env python3
"""
Script to convert Blender render data to H5 format.
This script processes the output from the Blender rendering pipeline and 
converts it to a more compact H5 format, with transparent backgrounds 
converted to white and objects cropped and scaled to fill the frame.
"""

import os
import glob
import argparse
import numpy as np
import h5py
import cv2
import OpenEXR
import Imath
from pathlib import Path
from tqdm import tqdm

def is_instance_valid(instance_path, min_valid_depth=5.0):
    """
    Check if an instance has valid depth data before processing.
    Looks at all depth files in the instance and returns True only if at least one has valid depth.
    
    Args:
        instance_path: Path to the instance directory
        min_valid_depth: Minimum depth threshold to consider depth valid
        
    Returns:
        bool: True if at least one depth file has valid depth, False otherwise
        str: reason for skipping if False
    """
    # Find all depth files
    depth_files = glob.glob(os.path.join(instance_path, "depth_*.exr"))
    
    if not depth_files:
        return False
    
    # Check each depth file
    for depth_file in depth_files:
        try:
            # Open the EXR file
            exr_file = OpenEXR.InputFile(depth_file)
            
            # Get dimensions
            dw = exr_file.header()['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # Try to read the 'V' channel 
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            try:
                depth_str = exr_file.channel('V', pixel_type)
            except:
                continue
            
            # Convert to numpy array
            depth = np.frombuffer(depth_str, dtype=np.float32).reshape(height, width)
            
            # Find min valid depth (ignoring zeros, nans, and infs)
            valid_mask = (depth > 0) & ~np.isnan(depth) & ~np.isinf(depth)
            if not np.any(valid_mask):
                continue
            
            min_depth = np.min(depth[valid_mask])
            max_depth = np.max(depth[valid_mask])
            
            # if there is one invalid depth file, we skip the instance
            if min_depth > min_valid_depth:
                return False
            if max_depth <= 0.:
                return False
                
        except Exception as e:
            return False
    
    # If we successfully get here, all depth files are valid
    return True

def process_exr_to_depth(exr_path, max_depth=100.0):
    """
    Convert EXR depth file to uint16 depth map in millimeters.
    Only uses channel 'V'.
    
    Args:
        exr_path: Path to the EXR depth file
        max_depth: Maximum depth in meters to consider valid (beyond this will be set to 0)
    
    Returns:
        numpy.ndarray: Depth map as uint16 in millimeters with invalid points set to 0
    """
    try:
        # Open the EXR file
        exr_file = OpenEXR.InputFile(exr_path)
        
        # Get the data window (size)
        dw = exr_file.header()['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Try to read only the 'V' channel 
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        try:
            depth_str = exr_file.channel('V', pixel_type)
        except:
            # If can't read 'V' channel, return zeros
            return np.zeros((height, width), dtype=np.uint16)
        
        # Convert to numpy array
        depth = np.frombuffer(depth_str, dtype=np.float32).reshape(height, width)
        
        # Filter out invalid values
        valid_mask = (depth > 0) & ~np.isnan(depth) & ~np.isinf(depth) & (depth <= max_depth)
        depth_filtered = np.zeros_like(depth)
        depth_filtered[valid_mask] = depth[valid_mask]
        
        # Convert to millimeters and to uint16
        depth_mm = (depth_filtered * 1000.0).astype(np.uint16)  # Convert to mm
        
        return depth_mm
        
    except Exception as e:
        # If any error occurs, return zeros
        return np.zeros((height, width), dtype=np.uint16)

def convert_rgba_to_rgb(rgba_image):
    """
    Convert RGBA image to RGB with white background.
    
    Args:
        rgba_image: RGBA image with shape (H, W, 4)
    
    Returns:
        numpy.ndarray: RGB image with shape (H, W, 3)
    """
    # Create a white background
    white_background = np.ones_like(rgba_image[:, :, :3]) * 255
    
    # Extract alpha channel and create 3-channel alpha for blending
    alpha = rgba_image[:, :, 3:4] / 255.0
    alpha_3 = np.repeat(alpha, 3, axis=2)
    
    # Blend foreground with white background based on alpha
    rgb = rgba_image[:, :, :3] * alpha_3 + white_background * (1 - alpha_3)
    
    return rgb.astype(np.uint8)

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

def crop_and_scale_image(rgb_img, depth_img, intrinsics, target_size=512):
    """
    Crop the image to the smallest square containing all non-white pixels,
    scale to target size, and adjust intrinsics accordingly.
    
    Args:
        rgb_img: RGB image as numpy array with shape (H, W, 3)
        depth_img: Depth image as numpy array with shape (H, W)
        intrinsics: 3x3 camera intrinsics matrix
        target_size: Target size for the output image (square)
        
    Returns:
        tuple: (cropped_rgb, cropped_depth, new_intrinsics, crop_params)
              where crop_params is (min_row, min_col, max_row, max_col, scale_factor)
    """
    # Find bounding box of non-white pixels
    min_row, min_col, max_row, max_col = find_object_bounding_box(rgb_img)
    
    # Make bounding box square
    min_row, min_col, max_row, max_col = make_square_bbox(
        min_row, min_col, max_row, max_col, rgb_img.shape[0], rgb_img.shape[1])
    
    # Crop RGB and depth images
    cropped_rgb = rgb_img[min_row:max_row+1, min_col:max_col+1]
    cropped_depth = depth_img[min_row:max_row+1, min_col:max_col+1]
    
    # Get current size and calculate scale factor
    current_size = cropped_rgb.shape[0]  # Should be square now
    scale_factor = target_size / current_size
    
    # Resize RGB image
    cropped_rgb = cv2.resize(cropped_rgb, (target_size, target_size), 
                             interpolation=cv2.INTER_LINEAR)
    
    # Resize depth image (nearest neighbor to avoid interpolation artifacts)
    cropped_depth = cv2.resize(cropped_depth, (target_size, target_size), 
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
    
    # Return cropped images, new intrinsics, and parameters for reference
    crop_params = (min_row, min_col, max_row, max_col, scale_factor)
    return cropped_rgb, cropped_depth, new_intrinsics, crop_params

def convert_to_h5(data_root, categories=None, target_size=512):
    """
    Convert Blender render data to H5 format with object-focused cropping and scaling.
    
    Args:
        input_dir: Root directory containing render data
        output_dir: Directory where H5 files will be saved
        categories: List of categories to process
        target_size: Target size for the cropped and scaled images (square)
    """
    input_dir = os.path.join(data_root, "render")
    output_dir = os.path.join(data_root, "h5")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    all_category_stats = {}

    # if no categories, get all categories in input_dir
    if categories is None:
        categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for category in categories:
        print(f"Processing category: {category}")
        category_path = os.path.join(input_dir, category)
        all_category_stats[category] = {
            "total_instances": 0,
            "valid_instances": 0,
            "has_rgb_no_depth": 0,
            "no_rgb_no_depth": 0,
        }
        
        # Skip if category directory doesn't exist
        if not os.path.isdir(category_path):
            print(f"Warning: Category directory {category} not found, skipping.")
            continue
        
        # Get all instance directories
        instance_dirs = sorted([d for d in os.listdir(category_path) 
                               if os.path.isdir(os.path.join(category_path, d))])
        
        if not instance_dirs:
            print(f"Warning: No instance directories found for {category}, skipping.")
            continue
        
        # Create H5 file for this category
        h5_path = os.path.join(output_dir, f"{category}.h5")
        with h5py.File(h5_path, 'w') as h5f:
            valid_instances = 0
            skipped_instances = 0
            
            for instance_idx in tqdm(instance_dirs, desc=f"Converting {category} instances"):
                instance_path = os.path.join(category_path, instance_idx)
                
                all_category_stats[category]["total_instances"] += 1
                is_valid = is_instance_valid(instance_path)
                if not is_valid: # skip if not valid. Collect stats
                    # check if rgba is all transparent
                    rgba_path = os.path.join(instance_path, "000.png")
                    # Load the RGBA image (this loads in BGRA format)
                    bgra_img = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
                    # Convert from BGRA to RGBA
                    rgba_img = cv2.cvtColor(bgra_img, cv2.COLOR_BGRA2RGBA)
                    if np.all(rgba_img[:, :, 3] == 0):
                        all_category_stats[category]["no_rgb_no_depth"] += 1
                    else:
                        all_category_stats[category]["has_rgb_no_depth"] += 1
                    skipped_instances += 1
                    continue
                
                # Get all PNG files (rendered images)
                png_files = sorted(glob.glob(os.path.join(instance_path, "*.png")))
                n_views = len(png_files)
                
                if n_views == 0:
                    print(f"  Skipping instance {instance_idx}: No PNG files found")
                    skipped_instances += 1
                    continue
                
                # Instance is valid - create group and process data
                all_category_stats[category]["valid_instances"] += 1
                instance_group = h5f.create_group(f"instance_{instance_idx}")
                valid_instances += 1
                
                # Read one image to get dimensions
                sample_img = cv2.imread(png_files[0], cv2.IMREAD_UNCHANGED)  # RGBA
                img_height, img_width = sample_img.shape[:2]
                
                # Read camera intrinsic matrix
                intrinsics_path = os.path.join(instance_path, "cam_K.npy")
                if os.path.exists(intrinsics_path):
                    intrinsics = np.load(intrinsics_path)
                else:
                    intrinsics = np.eye(3, dtype=np.float32)
                
                # Initialize arrays for all views - now with target_size dimensions
                rgb_data = np.zeros((n_views, target_size, target_size, 3), dtype=np.uint8)
                depth_data = np.zeros((n_views, target_size, target_size), dtype=np.uint16)
                extrinsics_data = np.zeros((n_views, 4, 4), dtype=np.float32)
                
                # Store adjusted intrinsics for each view
                adjusted_intrinsics_data = np.zeros((n_views, 3, 3), dtype=np.float32)
                
                # Process each view
                for view_idx, png_file in enumerate(png_files):
                    # Extract view number from filename
                    view_num = int(os.path.basename(png_file).split('.')[0])
                    
                    # Read and convert RGB image
                    rgba_img = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)  # RGBA
                    # Convert from BGRA to RGBA
                    rgba_img = cv2.cvtColor(rgba_img, cv2.COLOR_BGRA2RGBA)
                    rgb_img = convert_rgba_to_rgb(rgba_img)
                    
                    # Find and read corresponding depth file
                    depth_file = None
                    patterns = [
                        f"depth_{view_num:03d}*.exr",     # depth_000*.exr
                        f"depth_{view_num:04d}0001.exr",  # depth_0000001.exr 
                        f"depth_{view_num:07d}.exr"       # depth_0000000.exr
                    ]
                    
                    depth_img = np.zeros((img_height, img_width), dtype=np.uint16)
                    
                    for pattern in patterns:
                        matches = glob.glob(os.path.join(instance_path, pattern))
                        if matches:
                            depth_file = matches[0]
                            try:
                                depth_img = process_exr_to_depth(depth_file)
                            except Exception as e:
                                print(f"  Warning: Failed to process depth file {depth_file}, using zeros")
                            break
                    
                    # Read camera extrinsics
                    extrinsics_path = os.path.join(instance_path, f"{view_num:03d}.npy")
                    if os.path.exists(extrinsics_path):
                        # Load 3x4 RT matrix and convert to 4x4
                        rt_matrix = np.load(extrinsics_path)  # 3x4 matrix
                        extrinsic_matrix = np.eye(4, dtype=np.float32)
                        extrinsic_matrix[:3, :] = rt_matrix
                        extrinsics_data[view_idx] = extrinsic_matrix
                    else:
                        extrinsics_data[view_idx] = np.eye(4, dtype=np.float32)
                    
                    # Apply crop and scale
                    cropped_rgb, cropped_depth, adjusted_intrinsics, crop_params = crop_and_scale_image(
                        rgb_img, depth_img, intrinsics, target_size)
                    
                    # Store processed data
                    rgb_data[view_idx] = cropped_rgb
                    depth_data[view_idx] = cropped_depth
                    adjusted_intrinsics_data[view_idx] = adjusted_intrinsics
                
                # Store all data in the H5 file
                instance_group.create_dataset("rgb", data=rgb_data, 
                                             compression="gzip", compression_opts=6)
                instance_group.create_dataset("depth", data=depth_data, 
                                             compression="gzip", compression_opts=6)
                instance_group.create_dataset("extrinsics", data=extrinsics_data)
                instance_group.create_dataset("intrinsics", data=adjusted_intrinsics_data)
            
            print(f"Processed {valid_instances} valid instances, skipped {skipped_instances} invalid instances")
        
        print(f"Saved category {category} to {h5_path}")
    
    display_stats(all_category_stats)

def display_stats(all_category_stats):
    """
    Display stats for each category
    """
    print("\nSummary:")
    print("=" * 100)
    print(f"{'Category':<20} {'Total':<10} {'Valid':<10} {'Has RGB No Depth':<20} {'No RGB No Depth':<20} {'Valid Percentage':<20}")
    print("-" * 100)

    grand_total = 0
    grand_valid = 0
    grand_has_rgb_no_depth = 0
    grand_no_rgb_no_depth = 0

    for category, stats in all_category_stats.items():
        total = stats["total_instances"]
        valid = stats["valid_instances"]
        has_rgb_no_depth = stats["has_rgb_no_depth"]
        no_rgb_no_depth = stats["no_rgb_no_depth"]
        percentage = (valid / total) * 100 if total > 0 else 0
        grand_total += total
        grand_valid += valid
        grand_has_rgb_no_depth += has_rgb_no_depth
        grand_no_rgb_no_depth += no_rgb_no_depth
        print(f"{category:<20} {total:<10} {valid:<10} {has_rgb_no_depth:<20} {no_rgb_no_depth:<20} {percentage:.1f}%")
    
    grand_percentage = (grand_valid / grand_total) * 100 if grand_total > 0 else 0
    grand_percentage_has_rgb_no_depth = (grand_has_rgb_no_depth / grand_total) * 100 if grand_total > 0 else 0
    grand_percentage_no_rgb_no_depth = (grand_no_rgb_no_depth / grand_total) * 100 if grand_total > 0 else 0
    print("-" * 100)
    print(f"{'Grand Total':<20} {grand_total:<10} {grand_valid:<10} {grand_percentage_has_rgb_no_depth:.1f}% {grand_percentage_no_rgb_no_depth:.1f}% {grand_percentage:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Convert Blender render data to H5 format with object-focused cropping")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing render data")
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                        help="Categories to process. If not provided, all categories in input_dir will be processed.")
    parser.add_argument("--min_valid_depth", type=float, default=5.0,
                        help="Minimum depth threshold to consider an instance valid")
    parser.add_argument("--target_size", type=int, default=384,
                        help="Target size for cropped and scaled images (square)")
    
    args = parser.parse_args()
    
    print(f"Converting data from {args.data_root}/render to H5 format in {args.data_root}/h5")
    convert_to_h5(args.data_root, args.categories, args.target_size)
    print("Conversion complete!")

if __name__ == "__main__":
    main()