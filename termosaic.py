import argparse
import os
import shutil
from enum import Enum

import numpy as np
from PIL import Image


class Format(Enum):
    IMAGE = "i"
    VIDEO = "v"

    def valid(self):
        if self == Format.IMAGE:
            return {"img", "image", "i"}
        elif self == Format.VIDEO:
            return {"vid", "video", "v"}


class Colors(Enum):
    EXTENDED = "256"
    BASIC = "16"

    def valid(self):
        if self == Colors.EXTENDED:
            return {"256", "full", "all", "extended"}
        elif self == Colors.BASIC:
            return {"16", "half", "some", "basic"}


# Handle argument parsing


# Format type argument handler
def format_type(value):
    value_lower = value.lower()
    if value_lower in Format.IMAGE.valid():
        return Format.IMAGE
    elif value_lower in Format.VIDEO.valid():
        return Format.VIDEO
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid value for --format: '{value}'. Valid values are: {Format.IMAGE.valid()} for images and {Format.VIDEO.value()} for videos."
        )


# Colors type argument handler
def colors_type(value):
    value_lower = value.lower()
    if value_lower in Colors.EXTENDED.valid():
        return Colors.EXTENDED
    elif value_lower in Colors.BASIC.valid():
        return Colors.BASIC
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid value for --colors: '{value}'. Valid values are: {Colors.EXTENDED.valid()} for 256 colors or {Colors.BASIC.valid()} for 16 colors."
        )


parser = argparse.ArgumentParser(description="Print images on terminal")

parser.add_argument(
    "-f",
    "--format",
    type=format_type,
    required=True,
    help=(
        "Specify the format type. Valid values are case-insensitive: "
        f"{Format.IMAGE.valid()} for images or {Format.VIDEO.valid()} for videos."
    ),
)

parser.add_argument(
    "-c",
    "--colors",
    type=colors_type,
    help=(
        "Specify the color depth. Valid values are case-insensitive: "
        f"{Colors.EXTENDED.valid()} for 256 colors or {Colors.BASIC.valid()} for 16 colors."
    ),
    default="256",
)

parser.add_argument(
    "filename",
    help="Specify the filename (required positional argument).",
)

# Constants for basic 16 terminal color codes (ANSI)
BASIC_16_COLORS = {
    (0, 0, 0): "30",  # Black
    (128, 0, 0): "31",  # Red
    (0, 128, 0): "32",  # Green
    (128, 128, 0): "33",  # Yellow
    (0, 0, 128): "34",  # Blue
    (128, 0, 128): "35",  # Magenta
    (0, 128, 128): "36",  # Cyan
    (192, 192, 192): "37",  # Light Gray
    (128, 128, 128): "90",  # Dark Gray
    (255, 0, 0): "91",  # Bright Red
    (0, 255, 0): "92",  # Bright Green
    (255, 255, 0): "93",  # Bright Yellow
    (0, 0, 255): "94",  # Bright Blue
    (255, 0, 255): "95",  # Bright Magenta
    (0, 255, 255): "96",  # Bright Cyan
    (255, 255, 255): "97",  # White
}

# Generate a 256 color lookup table for ANSI escape codes (16 - 255)
EXTENDED_256_COLORS = {
    (r, g, b): str(16 + 36 * int(r / 51) + 6 * int(g / 51) + int(b / 51))
    for r in range(0, 256, 51)
    for g in range(0, 256, 51)
    for b in range(0, 256, 51)
}

BLOCK = chr(0x2588)


def is_terminal():
    return os.isatty(1)


def get_terminal_size():
    if not is_terminal():
        return None
    try:
        terminal_size = shutil.get_terminal_size(fallback=(80, 24))
        return terminal_size.columns, terminal_size.lines
    except Exception:
        raise Exception("cannot detect the terminal size")


def resize_image(image_path, max_width, max_height):
    # Resizes image maintaining aspect ratio
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
    except FileNotFoundError:
        raise Exception(f"Image file not found at {image_path}")

    original_width, original_height = img.size

    if original_width <= max_width and original_height <= max_height:
        return img

    width_ratio = max_width / original_width
    height_ratio = max_height / original_height
    scale_ratio = min(width_ratio, height_ratio)

    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    try:
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except AttributeError:
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_img


def map_to_nearest_colors(image_matrix, color_mode):
    color_map = EXTENDED_256_COLORS
    escape_prefix = "\033[38;5;"
    escape_suffix = f"m{BLOCK}{BLOCK}"

    if color_mode == Colors.BASIC:
        color_map = BASIC_16_COLORS
        escape_prefix = "\033["

    color_keys = np.array(list(color_map.keys()))  # Shape (N, 3)
    color_values = np.array(list(color_map.values()))  # Shape (N,)

    # Convert color values to ANSI escape codes
    color_values = np.char.add(np.char.add(escape_prefix, color_values), escape_suffix)

    # Reshape to (H*W, 3)
    h, w, _ = image_matrix.shape
    flat_pixels = image_matrix.reshape(-1, 3)  # Shape (H*W, 3)

    diffs = (
        flat_pixels[:, np.newaxis, :] - color_keys[np.newaxis, :, :]
    )  # Shape (H*W, N, 3)
    distances = np.sqrt(np.sum(diffs**2, axis=2))  # Shape (H*W, N)

    # Find min distance for each pixel
    min_indices = np.argmin(distances, axis=1)  # Shape (H*W,)

    # Map indices to color codes
    closest_colors = color_values[min_indices]  # Shape (H*W,)

    # Reshape back to (H, W)
    return closest_colors.reshape(h, w)


def print_matrix(color_matrix):
    joined_rows = np.apply_along_axis("".join, axis=1, arr=color_matrix)
    result = "\n".join(joined_rows)
    reset_color = "\033[0m"
    print(result, reset_color, sep="")


def process_image(image_path, color_mode):
    terminal_size = get_terminal_size()
    if terminal_size is None:
        print("Not a terminal, cannot process image for display.")
        return

    terminal_width, terminal_height = terminal_size
    max_image_width = terminal_width // 2
    max_image_height = terminal_height

    resized_image = resize_image(image_path, max_image_width, max_image_height)

    # numpy array (height, width, RGB)
    image_matrix_3d = np.array(resized_image)
    color_matrix_2d = map_to_nearest_colors(image_matrix_3d, color_mode)
    print_matrix(color_matrix_2d)


def main():
    args = parser.parse_args()
    filepath = args.filename
    color_mode_selected = args.colors
    file_format = args.format

    if file_format == Format.VIDEO:
        raise NotImplementedError("video support is not implemented yet.")

    process_image(filepath, color_mode=color_mode_selected)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)
