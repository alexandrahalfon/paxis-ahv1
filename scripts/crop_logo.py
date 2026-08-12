#!/usr/bin/env python3
"""Crop white borders from frontend/assets/logo.png and overwrite the file."""
from PIL import Image
import os

LOGO_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "assets", "logo.png")

def trim_white_simple(im, threshold=248):
    """Crop to bounding box where any channel is below threshold (not pure white)."""
    rgb = im.convert("RGB")
    data = rgb.load()
    w, h = im.size
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for y in range(h):
        for x in range(w):
            r, g, b = data[x, y]
            if r < threshold or g < threshold or b < threshold:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
    if x_min > x_max or y_min > y_max:
        return im
    return im.crop((x_min, y_min, x_max + 1, y_max + 1))

if __name__ == "__main__":
    path = os.path.abspath(LOGO_PATH)
    if not os.path.isfile(path):
        print("Logo not found:", path)
        exit(1)
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    else:
        im = im.copy()
    cropped = trim_white_simple(im, threshold=250)
    cropped.save(path, "PNG")
    print("Cropped and saved:", path)
