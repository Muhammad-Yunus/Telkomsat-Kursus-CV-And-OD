import argparse
from osgeo import gdal
import numpy as np
import cv2
import os

# --- Helper function for percentile stretch ---
def stretch_uint16_to_uint8(img, lower_percent=2, upper_percent=98, gamma=1.0):
    """
    Stretch a 16-bit image to 8-bit using percentile scaling with optional gamma correction.
    """
    # Compute percentiles
    low = np.percentile(img, lower_percent)
    high = np.percentile(img, upper_percent)

    # Clip and scale to 0-255
    stretched = np.clip((img - low) * 255.0 / (high - low), 0, 255)

    # Apply gamma correction
    if gamma != 1.0:
        stretched = ((stretched / 255.0) ** (1.0 / gamma)) * 255

    return stretched.astype(np.uint8)


# ============================
# Main Program
# ============================
def main():
    parser = argparse.ArgumentParser(description="Convert 16-bit GeoTIFF to 8-bit RGB TIFF")
    parser.add_argument("--input", required=True, help="Input 16-bit GeoTIFF")
    parser.add_argument("--output", required=True, help="Output filename for 8-bit TIFF")
    args = parser.parse_args()

    input_file_path = args.input
    output_file_path = args.output

    # Load image
    ds = gdal.Open(input_file_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open {input_file_path}")

    # Read bands (assuming B,G,R order)
    blue = ds.GetRasterBand(1).ReadAsArray()
    green = ds.GetRasterBand(2).ReadAsArray()
    red = ds.GetRasterBand(3).ReadAsArray()

    # Apply stretch
    gamma_value = 1.2
    blue_stretch = stretch_uint16_to_uint8(blue, gamma=gamma_value)
    green_stretch = stretch_uint16_to_uint8(green, gamma=gamma_value)
    red_stretch = stretch_uint16_to_uint8(red, gamma=gamma_value)

    # Merge into BGR image for OpenCV
    bgr_img = cv2.merge([blue_stretch, green_stretch, red_stretch])
    # Save as 8-bit TIFF
    cv2.imwrite(output_file_path, bgr_img)

    print(f"Saved 8-bit image as: {output_file_path}")
    print("Image shape:", bgr_img.shape)
    print("Image dtype:", bgr_img.dtype)
    print("Number of bands:", ds.RasterCount)


if __name__ == "__main__":
    main()
