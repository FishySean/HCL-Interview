"""Real fast-tier detector: a lightweight single-stage person detector.

Why this model class (the ML decision, stated where the code lives):

* Person presence is a *detection* problem, not classification. A detector
  returns a box, which gives a distance proxy (box area) and, with a tracker,
  identity continuity -- so the robot greets a person, not a frame.
* "Person" is COCO class 0, the best-represented class in public detection
  data, so off-the-shelf accuracy is high with zero data collection.
* YOLO11n / MobileNet-SSD class models run >30 FPS on modest CPU/edge hardware,
  quantize to INT8, and deploy through ONNX Runtime / TFLite / CoreML.

Alternatives considered and rejected: background subtraction (a motionless
person vanishes), Haar/HOG (poor recall on side and partial views), a face
detector as the *primary* signal (a person facing away is still present), and a
VLM in this hot path (hundreds of milliseconds, non-deterministic, costly --
it belongs in the slow tier instead).

The heavy dependencies are imported lazily so the core runtime, the demo, and
the entire test suite stay dependency-free.
"""

from __future__ import annotations

from ...contracts.events import PersonObservation

_INSTALL_HINT = (
    "The camera detector needs optional extras. Install them with:\n"
    "    pip install -r requirements-camera.txt\n"
    "The simulated detectors require nothing and satisfy the same protocol."
)


class YoloPersonDetector:
    """Wraps Ultralytics' tracking API; `persist=True` keeps track ids stable
    across frames, which is what lets the debouncer reason about one person."""

    name = "yolo"
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        source: int | str = 0,
        model_name: str = "yolo11n.pt",
        min_confidence: float = 0.5,
        imgsz: int = 480,
    ) -> None:
        try:
            import cv2  # noqa: PLC0415
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(_INSTALL_HINT) from exc

        self._cv2 = cv2
        self._min_confidence = min_confidence
        self._imgsz = imgsz
        self._model = YOLO(model_name)
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():  # pragma: no cover - depends on hardware
            raise RuntimeError(f"could not open video source {source!r}")

    async def detect(self) -> PersonObservation:
        ok, frame = self._capture.read()
        if not ok:  # pragma: no cover - depends on hardware
            return PersonObservation(present=False, source=self.name)

        results = self._model.track(
            frame,
            persist=True,
            classes=[self.PERSON_CLASS_ID],
            conf=self._min_confidence,
            imgsz=self._imgsz,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return PersonObservation(present=False, source=self.name)

        # The closest person drives the interaction; box area is the distance proxy.
        best = max(range(len(boxes)), key=lambda i: _area(boxes.xyxy[i].tolist()))
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[best].tolist())
        track_ids = boxes.id
        return PersonObservation(
            present=True,
            confidence=float(boxes.conf[best]),
            bbox=(x1, y1, x2, y2),
            track_id=int(track_ids[best]) if track_ids is not None else None,
            source=self.name,
        )

    async def close(self) -> None:
        self._capture.release()


def _area(xyxy: list[float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
