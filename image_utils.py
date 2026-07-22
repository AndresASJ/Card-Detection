"""Image loading, preprocessing, and perspective correction."""

from pathlib import Path

import cv2
import numpy as np


CARD_WIDTH = 300
CARD_HEIGHT = 420
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def display_name(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ").title()
    return name.replace(" Of ", " of ")


def load_images_from_directory(directory: str | Path) -> list[tuple[str, np.ndarray]]:
    """Load normalized grayscale card templates from a directory."""
    template_dir = Path(directory).expanduser()
    if not template_dir.exists():
        return []

    images: list[tuple[str, np.ndarray]] = []
    for image_path in sorted(template_dir.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        normalized = cv2.resize(
            image,
            (CARD_WIDTH, CARD_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        images.append((display_name(image_path), normalized))
    return images


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Produce a foreground mask for light cards on a contrasting surface."""
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    border = np.concatenate(
        (binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1])
    )
    if float(np.mean(border)) > 127:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)


def order_points(points: np.ndarray) -> np.ndarray:
    """Return four points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sums = pts.sum(axis=1)
    point_differences = np.diff(pts, axis=1).reshape(4)
    ordered[0] = pts[np.argmin(point_sums)]
    ordered[2] = pts[np.argmax(point_sums)]
    ordered[1] = pts[np.argmin(point_differences)]
    ordered[3] = pts[np.argmax(point_differences)]
    return ordered


def warp_card(
    frame: np.ndarray,
    corners: np.ndarray,
    width: int = CARD_WIDTH,
    height: int = CARD_HEIGHT,
) -> np.ndarray:
    """Flatten a detected quadrilateral into a portrait card image."""
    source = order_points(corners)
    top_width = np.linalg.norm(source[1] - source[0])
    bottom_width = np.linalg.norm(source[2] - source[3])
    left_height = np.linalg.norm(source[3] - source[0])
    right_height = np.linalg.norm(source[2] - source[1])

    if (top_width + bottom_width) > (left_height + right_height):
        source = source[[3, 0, 1, 2]]

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(frame, matrix, (width, height))
