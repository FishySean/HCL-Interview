from .simulated import ManualDetector, ScriptedDetector, Segment

__all__ = ["ManualDetector", "ScriptedDetector", "Segment"]


def load_yolo_detector(**kwargs):
    """Imported lazily: the camera path must never be a cost for the core."""
    from .yolo import YoloPersonDetector

    return YoloPersonDetector(**kwargs)
