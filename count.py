import cv2
import glob
import numpy as np

def main():
    mask_paths = glob.glob(r'C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\masks\*.png')

    total_background_pixels = 0
    total_tree_pixels = 0
    count_with_background = 0
    count_with_tree = 0

    tree_ratios = []
    heights = []
    widths = []

    for path in mask_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 0).astype(np.uint8)

        bg_pixels = np.sum(binary_mask == 0)
        tree_pixels = np.sum(binary_mask == 1)

        total_background_pixels += bg_pixels
        total_tree_pixels += tree_pixels

        if bg_pixels > 0:
            count_with_background += 1
        if tree_pixels > 0:
            count_with_tree += 1

        tree_ratio = tree_pixels / (bg_pixels + tree_pixels)
        tree_ratios.append(tree_ratio)

        h, w = binary_mask.shape
        heights.append(h)
        widths.append(w)

    print("----- Dataset Statistics Report -----")
    print(f"Total number of images: {len(mask_paths)}")
    print(f"Number of images containing background: {count_with_background} / {len(mask_paths)}")
    print(f"Number of images containing trees: {count_with_tree} / {len(mask_paths)}")
    print()
    print(f"Total background pixels: {total_background_pixels}")
    print(f"Total tree pixels: {total_tree_pixels}")
    print()
    print(f"Mean tree pixel ratio: {np.mean(tree_ratios):.4f}")
    print(f"Maximum tree pixel ratio: {np.max(tree_ratios):.4f}")
    print(f"Minimum tree pixel ratio: {np.min(tree_ratios):.4f}")
    print()
    print(f"Mean image height: {np.mean(heights):.1f}, Max: {np.max(heights)}, Min: {np.min(heights)}")
    print(f"Mean image width: {np.mean(widths):.1f}, Max: {np.max(widths)}, Min: {np.min(widths)}")

if __name__ == "__main__":
    main()
