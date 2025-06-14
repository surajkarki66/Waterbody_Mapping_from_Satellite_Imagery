import os
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm

def normalize(array):
    """Normalize array to 0–255 and convert to uint8."""
    array = array.astype(np.float32)
    if array.max() == array.min():
        return np.zeros_like(array, dtype=np.uint8)
    return ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

def convert_tif_to_png(base_dir):
    for split in ['train', 'val', 'test']:
        for subfolder in ['img', 'msk']:
            tif_folder = os.path.join(base_dir, split, subfolder)
            png_folder = tif_folder.replace(base_dir, base_dir + "_png")
            os.makedirs(png_folder, exist_ok=True)

            tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]

            for tif_file in tqdm(tif_files, desc=f"Converting {split}/{subfolder}"):
                tif_path = os.path.join(tif_folder, tif_file)
                png_path = os.path.join(png_folder, tif_file.replace('.tif', '.png'))

                with rasterio.open(tif_path) as src:
                    if subfolder == 'img':
                        # Read RGB: Band 3 (Red), Band 2 (Green), Band 1 (Blue)
                        red = normalize(src.read(3))
                        green = normalize(src.read(2))
                        blue = normalize(src.read(1))
                        rgb = np.dstack([red, green, blue])
                        Image.fromarray(rgb).save(png_path)
                    else:
                        # For masks (assuming single-band with values 0 and 1)
                        mask = src.read(1)
                        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_img.save(png_path)

if __name__ == "__main__":
    convert_tif_to_png('./output_data')  # Root folder containing train/ val/ test/

