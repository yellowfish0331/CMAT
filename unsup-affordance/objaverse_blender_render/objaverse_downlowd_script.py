import objaverse
import os
import shutil


def download_category(data_root, category, category_uids, n=None):
    """
    Downloads N random models from the specified category.
    """
    if n is not None and len(category_uids) > n:
        category_uids = category_uids[:n]

    category_objs = objaverse.load_objects(category_uids)

    category_root = f"{data_root}/obj_models/{category}"
    os.makedirs(category_root, exist_ok=True)

    idx = 0
    for key, value in category_objs.items():
        shutil.move(value, os.path.join(category_root, f"{idx}.glb"))
        idx += 1

    print(f"Downloaded {idx} models for category: {category}")


def main(data_root, categories, n):

    lvis_annotations = objaverse.load_lvis_annotations()

    for category in categories:
        category_uids = lvis_annotations.get(category, [])
        if len(category_uids) == 0: continue
        download_category(data_root, category, category_uids, n=n)


if __name__ == "__main__":

    from used_categories import CATEGORIES
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='path to save the downloaded data')
    parser.add_argument('--n', type=int, default=50, help='number of models to download for each category')
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    main(args.data_root, CATEGORIES, args.n)

