"""Real-time playing-card detection and ORB recognition."""

import argparse
import re
from datetime import datetime
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from card import Card
from card_detector import detect_cards
from image_utils import (
    CARD_HEIGHT,
    CARD_WIDTH,
    load_images_from_directory,
    preprocess_frame,
)
from template_matcher import TemplateMatcher


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "templates"
DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "captures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and identify playing cards from a webcam feed."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--templates",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        help="Directory containing captured card templates",
    )
    parser.add_argument(
        "--capture-template",
        metavar="NAME",
        help="Capture one normalized template using the webcam",
    )
    parser.add_argument("--debug", action="store_true", help="Show processing stages")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
        help="Directory used when P saves a frame",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.04,
        help="Minimum normalized ORB score for a known card",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=0,
        help="Automatically record a silent labeled clip, then exit",
    )
    parser.add_argument(
        "--record-output",
        type=Path,
        default=DEFAULT_CAPTURE_DIR / "card-detection-demo.mp4",
        help="MP4 destination used with --record-seconds",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Seconds to wait before automatic recording starts",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def open_camera(index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera {index}. Check macOS camera permission or try --camera 1."
        )
    return capture


def capture_template(args: argparse.Namespace) -> int:
    args.templates.mkdir(parents=True, exist_ok=True)
    destination = args.templates / f"{slugify(args.capture_template)}.png"
    capture = open_camera(args.camera)
    print("Place one card on a dark surface. Press Space to save; Q cancels.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")
            binary = preprocess_frame(frame)
            cards = detect_cards(binary, frame)
            preview = frame.copy()
            for card in cards:
                cv2.drawContours(preview, [card.contour], -1, (0, 200, 90), 3)
            cv2.putText(
                preview,
                f"Capture: {args.capture_template} | Space saves | Q cancels",
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 90),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Card Detection - Template Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 1
            if key == 32:
                if len(cards) != 1:
                    print(f"Expected one detected card; found {len(cards)}. Try again.")
                    continue
                cv2.imwrite(str(destination), cards[0].image)
                print(f"Saved {destination}")
                return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_debug_view(
    annotated: np.ndarray,
    binary: np.ndarray,
    cards: list[Card],
) -> np.ndarray:
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary_bgr = cv2.resize(binary_bgr, (annotated.shape[1], annotated.shape[0]))
    top = np.hstack((annotated, binary_bgr))

    strip = np.full((CARD_HEIGHT, top.shape[1], 3), 28, dtype=np.uint8)
    x = 18
    for card in cards[:4]:
        crop = cv2.resize(card.image, (CARD_WIDTH, CARD_HEIGHT))
        if x + CARD_WIDTH > strip.shape[1]:
            break
        strip[:, x : x + CARD_WIDTH] = crop
        cv2.putText(
            strip,
            card.identity,
            (x + 8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 90),
            2,
            cv2.LINE_AA,
        )
        x += CARD_WIDTH + 18
    return np.vstack((top, strip))


def save_capture(
    output_dir: Path,
    annotated: np.ndarray,
    debug_view: np.ndarray | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    annotated_path = output_dir / f"card-detection-{timestamp}.png"
    cv2.imwrite(str(annotated_path), annotated)
    print(f"Saved {annotated_path}")
    if debug_view is not None:
        debug_path = output_dir / f"card-detection-debug-{timestamp}.png"
        cv2.imwrite(str(debug_path), debug_view)
        print(f"Saved {debug_path}")


def run_recognition(args: argparse.Namespace) -> int:
    templates = load_images_from_directory(args.templates)
    if not templates:
        print(
            f"No templates found in {args.templates}. Capture one with "
            "--capture-template \"Ace of Spades\"."
        )
        return 2

    matcher = TemplateMatcher(templates, min_score=args.min_score)
    if not matcher.templates:
        print("The template images did not contain enough visual detail for ORB matching.")
        return 2

    print(f"Loaded {len(matcher.templates)} templates from {args.templates}")
    print("Press P to save the current frame and Q to quit.")
    capture = open_camera(args.camera)
    writer: cv2.VideoWriter | None = None
    recording_armed_at: float | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")
            binary = preprocess_frame(frame)
            cards = detect_cards(binary, frame)
            annotated = frame.copy()
            for card in cards:
                matcher.identify(card)
                card.draw_on_frame(annotated)

            if args.record_seconds > 0:
                if recording_armed_at is None:
                    cv2.putText(
                        annotated,
                        "Press Space to start recording",
                        (24, 44),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 200, 90),
                        3,
                        cv2.LINE_AA,
                    )
                else:
                    elapsed = monotonic() - recording_armed_at
                    remaining = args.countdown - elapsed
                    if remaining > 0:
                        cv2.putText(
                            annotated,
                            f"Recording in {max(1, int(remaining) + 1)}",
                            (24, 44),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 200, 90),
                            3,
                            cv2.LINE_AA,
                        )
                    else:
                        if writer is None:
                            args.record_output.parent.mkdir(parents=True, exist_ok=True)
                            height, width = annotated.shape[:2]
                            writer = cv2.VideoWriter(
                                str(args.record_output),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                12.0,
                                (width, height),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(
                                    f"Could not open video writer for {args.record_output}."
                                )
                            print(f"Recording {args.record_seconds:g} seconds...")
                        writer.write(annotated)
                        if elapsed >= args.countdown + args.record_seconds:
                            print(f"Saved {args.record_output}")
                            return 0

            cv2.imshow("Card Detection", annotated)
            debug_view = build_debug_view(annotated, binary, cards) if args.debug else None
            if debug_view is not None:
                cv2.imshow("Card Detection - Debug", debug_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == 32 and args.record_seconds > 0 and recording_armed_at is None:
                recording_armed_at = monotonic()
                print(f"Recording starts in {args.countdown} seconds...")
            if key == ord("p"):
                save_capture(args.save_dir, annotated, debug_view)
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    try:
        if args.capture_template:
            return capture_template(args)
        return run_recognition(args)
    except RuntimeError as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
