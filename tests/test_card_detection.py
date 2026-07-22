import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from card_detector import detect_cards, is_card_shaped
from image_utils import CARD_HEIGHT, CARD_WIDTH, load_images_from_directory, order_points, warp_card
from template_matcher import TemplateMatcher


def patterned_card(text: str = "ACE") -> np.ndarray:
    image = np.full((CARD_HEIGHT, CARD_WIDTH), 245, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (CARD_WIDTH - 9, CARD_HEIGHT - 9), 20, 5)
    cv2.putText(
        image,
        text,
        (35, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        25,
        4,
        cv2.LINE_AA,
    )
    cv2.circle(image, (150, 260), 55, 40, 6)
    cv2.line(image, (65, 355), (235, 315), 30, 7)
    return image


class GeometryTests(unittest.TestCase):
    def test_order_points(self) -> None:
        unordered = np.array([[90, 80], [10, 10], [10, 80], [90, 10]])
        expected = np.array([[10, 10], [90, 10], [90, 80], [10, 80]], dtype=np.float32)
        np.testing.assert_array_equal(order_points(unordered), expected)

    def test_warp_card_returns_normalized_portrait(self) -> None:
        frame = np.zeros((520, 620, 3), dtype=np.uint8)
        corners = np.array([[155, 70], [440, 105], [410, 465], [120, 430]])
        cv2.fillConvexPoly(frame, corners, (235, 235, 235))
        warped = warp_card(frame, corners)
        self.assertEqual(warped.shape, (CARD_HEIGHT, CARD_WIDTH, 3))
        self.assertGreater(float(warped.mean()), 200)

    def test_card_shape_filter(self) -> None:
        rectangle = np.array([[[0, 0]], [[200, 0]], [[200, 300]], [[0, 300]]])
        triangle = np.array([[[0, 0]], [[200, 0]], [[100, 300]]])
        tiny = np.array([[[0, 0]], [[20, 0]], [[20, 30]], [[0, 30]]])
        self.assertTrue(is_card_shaped(rectangle, min_area=1_000))
        self.assertFalse(is_card_shaped(triangle, min_area=1_000))
        self.assertFalse(is_card_shaped(tiny, min_area=1_000))

    def test_detect_cards_creates_normalized_crop(self) -> None:
        frame = np.zeros((500, 700, 3), dtype=np.uint8)
        corners = np.array([[180, 60], [430, 80], [410, 440], [160, 420]])
        cv2.fillConvexPoly(frame, corners, (245, 245, 245))
        binary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cards = detect_cards(binary, frame, min_area=10_000)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].image.shape, (CARD_HEIGHT, CARD_WIDTH, 3))


class TemplateTests(unittest.TestCase):
    def test_template_loading_uses_human_readable_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ace-of-spades.png"
            cv2.imwrite(str(path), patterned_card())
            templates = load_images_from_directory(directory)
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0][0], "Ace of Spades")
        self.assertEqual(templates[0][1].shape, (CARD_HEIGHT, CARD_WIDTH))

    def test_matcher_accepts_known_template(self) -> None:
        template = patterned_card("ACE")
        matcher = TemplateMatcher(
            [("Ace of Spades", template)],
            min_good_matches=4,
            min_score=0.01,
            min_margin=0.0,
        )
        result = matcher.match(template.copy())
        self.assertEqual(result.identity, "Ace of Spades")
        self.assertGreaterEqual(result.good_matches, 4)

    def test_matcher_accepts_rotated_template(self) -> None:
        template = patterned_card("ACE")
        matcher = TemplateMatcher(
            [("Ace of Spades", template)],
            min_good_matches=4,
            min_score=0.01,
            min_margin=0.0,
        )
        matrix = cv2.getRotationMatrix2D(
            (CARD_WIDTH / 2, CARD_HEIGHT / 2), 30, 0.85
        )
        rotated = cv2.warpAffine(
            template,
            matrix,
            (CARD_WIDTH, CARD_HEIGHT),
            borderValue=255,
        )
        result = matcher.match(rotated)
        self.assertEqual(result.identity, "Ace of Spades")

    def test_matcher_rejects_unknown_image(self) -> None:
        matcher = TemplateMatcher([("Ace of Spades", patterned_card("ACE"))])
        unknown = np.full((CARD_HEIGHT, CARD_WIDTH), 127, dtype=np.uint8)
        result = matcher.match(unknown)
        self.assertIsNone(result.identity)
        self.assertEqual(result.good_matches, 0)


if __name__ == "__main__":
    unittest.main()
