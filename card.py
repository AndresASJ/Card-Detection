"""Detected playing-card model and drawing helpers."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Card:
    contour: np.ndarray
    corners: np.ndarray
    image: np.ndarray
    identity: str = "Unknown"
    confidence: float = 0.0
    good_matches: int = 0

    def update_identity(
        self,
        identity: str | None,
        confidence: float,
        good_matches: int = 0,
    ) -> None:
        self.identity = identity or "Unknown"
        self.confidence = confidence
        self.good_matches = good_matches

    def draw_on_frame(self, frame: np.ndarray) -> None:
        color = (0, 200, 90) if self.identity != "Unknown" else (0, 180, 255)
        cv2.drawContours(frame, [self.contour], -1, color, 3)

        x, y, width, height = cv2.boundingRect(self.contour)
        label = (
            self.identity
            if self.identity == "Unknown"
            else f"{self.identity}  {self.confidence:.2f}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.62
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, scale, thickness
        )
        while text_width + 12 > width and scale > 0.42:
            scale -= 0.04
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, scale, thickness
            )

        label_x = x + 6
        label_y = y + height - baseline - 8
        cv2.rectangle(
            frame,
            (x, label_y - text_height - baseline - 6),
            (min(x + width, x + text_width + 12), y + height),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
