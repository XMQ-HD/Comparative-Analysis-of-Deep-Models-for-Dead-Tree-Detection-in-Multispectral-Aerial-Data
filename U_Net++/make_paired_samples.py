import os
import pickle

# Set directories for RGB, NIR, and mask images
rgb_dir = r"C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\RGB_images"
nir_dir = r"C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\NRG_images"
mask_dir = r"C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\masks"

samples = []
for fname in sorted(os.listdir(rgb_dir)):
    if not fname.lower().endswith('.png'):
        continue
    # Remove 'RGB_' prefix to get the basename
    basename = fname[len("RGB_"):]

    # Construct the corresponding NIR and mask filenames
    nir_fname = "NRG_" + basename
    mask_fname = "mask_" + basename

    rgb_path = os.path.join(rgb_dir, fname)
    nir_path = os.path.join(nir_dir, nir_fname)
    mask_path = os.path.join(mask_dir, mask_fname)
    if os.path.exists(nir_path) and os.path.exists(mask_path):
        samples.append((rgb_path, nir_path, mask_path))
    else:
        # Warning if the NIR or mask file is missing
        print(f"[Warning] Missing: {nir_fname if not os.path.exists(nir_path) else ''} {mask_fname if not os.path.exists(mask_path) else ''}")

# Save the paired sample paths to a pickle file
with open("paired_samples.pkl", "wb") as f:
    pickle.dump(samples, f)

print(f"✅ Found {len(samples)} samples in total. Saved to paired_samples.pkl")
