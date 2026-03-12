#!/usr/bin/env python3
"""
Texture Filter Script

This script processes GLB files in a directory structure, keeps only those with textures,
and ensures continuous indexing of the files.

Usage:
    blender --background --python texture_filter.py

The script will:
1. Go through each category in YOUR_DATA_DIR/obj_models/
2. Check each GLB file for textures
3. Keep only the models with textures
4. Re-index the remaining files to ensure continuous numbering
"""

import os
import sys
import bpy
import shutil
from pathlib import Path

def clear_scene():
    """Clear the current Blender scene."""
    # Delete all meshes, materials, etc.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for image in bpy.data.images:
        bpy.data.images.remove(image)

def analyze_model_textures(model_path):
    """Analyze texture information for a specific GLB model.
    
    Args:
        model_path (str): Path to the GLB model file
        
    Returns:
        bool: True if the model has textures, False otherwise
    """
    clear_scene()
    
    # Load the model
    try:
        bpy.ops.import_scene.gltf(filepath=model_path, merge_vertices=True)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return False
    
    # Check for textures
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
            
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                # Found a valid texture
                return True
    
    # No textures found
    return False

def process_category(category_path):
    """Process all GLB files in a category directory.
    
    Args:
        category_path (str): Path to the category directory
        
    Returns:
        tuple: (total_count, kept_count) - The total number of files and the number kept
    """
    category_dir = Path(category_path)
    
    if not category_dir.is_dir():
        print(f"Error: {category_path} is not a directory")
        return 0, 0
    
    # Create a temporary directory
    temp_dir = category_dir.parent / f"{category_dir.name}_temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Get all GLB files
    glb_files = sorted(list(category_dir.glob("*.glb")))
    total_count = len(glb_files)
    
    if total_count == 0:
        print(f"No GLB files found in {category_path}")
        return 0, 0
    
    print(f"Processing {category_dir.name}: {total_count} GLB files found")
    
    # Check each file and move those with textures to the temp directory with new indices
    kept_count = 0
    for file_path in glb_files:
        print(f"  Checking {file_path.name}...", end="", flush=True)
        
        if analyze_model_textures(str(file_path)):
            # Has texture, move to temp with new index
            new_file_name = f"{kept_count}.glb"
            shutil.copy2(file_path, temp_dir / new_file_name)
            kept_count += 1
            print(" ✓ Has texture")
        else:
            print(" ✗ No texture")
    
    # Now replace the original directory with the temp one
    for file_path in category_dir.glob("*.glb"):
        file_path.unlink()  # Remove original files
        
    # Copy from temp back to original directory
    for file_path in temp_dir.glob("*.glb"):
        shutil.copy2(file_path, category_dir / file_path.name)
        
    # Remove temp directory
    shutil.rmtree(temp_dir)
    
    return total_count, kept_count

def main(data_root):
    """Main function to process all categories."""
    # Base directory containing all categories
    base_dir = Path(data_root)
    
    # Make sure base directory exists
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        return
    
    # Get all category directories
    category_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if len(category_dirs) == 0:
        print(f"No category directories found in {base_dir}")
        return
    
    print(f"Found {len(category_dirs)} categories to process")
    
    # Process each category
    total_results = {}
    for category_dir in category_dirs:
        total, kept = process_category(category_dir)
        total_results[category_dir.name] = (total, kept)
    
    # Summary
    print("\nSummary:")
    print("=" * 50)
    print(f"{'Category':<20} {'Total':<10} {'Kept':<10} {'Percentage':<10}")
    print("-" * 50)
    
    grand_total = 0
    grand_kept = 0
    
    for category, (total, kept) in total_results.items():
        percentage = (kept / total) * 100 if total > 0 else 0
        print(f"{category:<20} {total:<10} {kept:<10} {percentage:.1f}%")
        grand_total += total
        grand_kept += kept
    
    print("-" * 50)
    grand_percentage = (grand_kept / grand_total) * 100 if grand_total > 0 else 0
    print(f"{'TOTAL':<20} {grand_total:<10} {grand_kept:<10} {grand_percentage:.1f}%")
    print("=" * 50)

# Optional: You can define specific categories to process
# CATEGORIES = ["chair", "table", "lamp"]
# If defined, modify the main function to use only these categories

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='path to the data root')
    args = parser.parse_args()
    main(args.data_root)