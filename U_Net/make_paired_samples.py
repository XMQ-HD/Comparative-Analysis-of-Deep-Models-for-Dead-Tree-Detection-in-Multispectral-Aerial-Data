import os
import pickle

# Set directory paths for RGB, NRG, and mask images
rgb_dir = r"C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\RGB_images"
nir_dir = r"C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\NRG_images"
mask_dir = r"C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\masks"

samples = []  # To store (rgb, nrg, mask) path tuples

# Iterate over all RGB images
for fname in sorted(os.listdir(rgb_dir)):
    if not fname.lower().endswith('.png'):
        continue
    # Remove "RGB_" prefix to get the base filename
    basename = fname[len("RGB_"):]
    # Construct corresponding NRG and mask filenames
    nir_fname = "NRG_" + basename
    mask_fname = "mask_" + basename

    rgb_path = os.path.join(rgb_dir, fname)
    nir_path = os.path.join(nir_dir, nir_fname)
    mask_path = os.path.join(mask_dir, mask_fname)

    # Only add samples if both NRG and mask exist
    if os.path.exists(nir_path) and os.path.exists(mask_path):
        samples.append((rgb_path, nir_path, mask_path))
    else:
        print(f"[Warning] Missing: {nir_fname if not os.path.exists(nir_path) else ''} {mask_fname if not os.path.exists(mask_path) else ''}")

# Save all paired samples to pickle
with open("paired_samples.pkl", "wb") as f:
    pickle.dump(samples, f)

print(f"✅ Found {len(samples)} samples in total. Saved to paired_samples.pkl")
