"""ORB-based card-template matching with unknown-card rejection."""

from dataclasses import dataclass

import cv2
import numpy as np

from card import Card


@dataclass(frozen=True)
class TemplateFeatures:
    name: str
    image: np.ndarray
    keypoints: tuple
    descriptors: np.ndarray


@dataclass(frozen=True)
class MatchResult:
    identity: str | None
    score: float
    good_matches: int
    margin: float


class TemplateMatcher:
    def __init__(
        self,
        templates: list[tuple[str, np.ndarray]],
        min_good_matches: int = 8,
        min_score: float = 0.04,
        min_margin: float = 0.008,
    ) -> None:
        self.orb = cv2.ORB_create(nfeatures=2_000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.min_good_matches = min_good_matches
        self.min_score = min_score
        self.min_margin = min_margin
        self.templates = self._prepare_templates(templates)

    def _prepare_templates(
        self,
        templates: list[tuple[str, np.ndarray]],
    ) -> list[TemplateFeatures]:
        prepared: list[TemplateFeatures] = []
        for name, image in templates:
            keypoints, descriptors = self.orb.detectAndCompute(image, None)
            if descriptors is None or not keypoints:
                continue
            prepared.append(
                TemplateFeatures(name, image, tuple(keypoints), descriptors)
            )
        return prepared

    def match(self, card_image: np.ndarray) -> MatchResult:
        grayscale = (
            cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
            if card_image.ndim == 3
            else card_image
        )
        keypoints, descriptors = self.orb.detectAndCompute(grayscale, None)
        if descriptors is None or not keypoints or not self.templates:
            return MatchResult(None, 0.0, 0, 0.0)

        candidates: list[tuple[float, int, str]] = []
        for template in self.templates:
            pairs = self.matcher.knnMatch(template.descriptors, descriptors, k=2)
            good = [
                first
                for pair in pairs
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < 0.75 * second.distance
            ]
            denominator = max(1, min(len(template.keypoints), len(keypoints)))
            candidates.append((len(good) / denominator, len(good), template.name))

        candidates.sort(reverse=True)
        best_score, best_count, best_name = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        margin = best_score - second_score
        accepted = (
            best_count >= self.min_good_matches
            and best_score >= self.min_score
            and margin >= self.min_margin
        )
        return MatchResult(
            best_name if accepted else None,
            best_score,
            best_count,
            margin,
        )

    def identify(self, card: Card) -> MatchResult:
        result = self.match(card.image)
        card.update_identity(result.identity, result.score, result.good_matches)
        return result


def identify_card(card: Card, matcher: TemplateMatcher) -> MatchResult:
    """Compatibility wrapper for callers that operate on Card objects."""
    return matcher.identify(card)
