"""Contour-based playing-card detection."""

import cv2
import numpy as np

from card import Card
from image_utils import warp_card


def approximate_card_contour(
    contour: np.ndarray,
    min_area: float = 5_000,
) -> np.ndarray | None:
    """Return a four-corner approximation when a contour resembles a card."""
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approximation) != 4 or not cv2.isContourConvex(approximation):
        return None

    _, _, width, height = cv2.boundingRect(approximation)
    if width == 0 or height == 0:
        return None
    aspect_ratio = width / height
    if not (0.45 <= aspect_ratio <= 0.90 or 1.10 <= aspect_ratio <= 2.20):
        return None

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0 or area / hull_area < 0.88:
        return None
    return approximation.reshape(4, 2)


def is_card_shaped(contour: np.ndarray, min_area: float = 5_000) -> bool:
    return approximate_card_contour(contour, min_area) is not None


def create_card(contour: np.ndarray, corners: np.ndarray, frame: np.ndarray) -> Card:
    return Card(contour=contour, corners=corners, image=warp_card(frame, corners))


def detect_cards(
    binary: np.ndarray,
    frame: np.ndarray,
    min_area: float | None = None,
) -> list[Card]:
    """Detect and flatten card-shaped contours, largest first."""
    frame_area = frame.shape[0] * frame.shape[1]
    effective_min_area = (
        min_area if min_area is not None else max(5_000, frame_area * 0.015)
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cards: list[Card] = []
    for contour in contours:
        corners = approximate_card_contour(contour, effective_min_area)
        if corners is not None:
            cards.append(create_card(contour, corners, frame))
    cards.sort(key=lambda card: cv2.contourArea(card.contour), reverse=True)
    return cards
