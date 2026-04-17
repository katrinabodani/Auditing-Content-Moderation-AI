"""
pipeline.py
-----------
Production-grade three-layer content moderation guardrail pipeline.

Layers:
    1. Input Filter  — fast regex pre-filter for obvious violations
    2. Calibrated Model — DistilBERT with isotonic probability calibration
    3. Human Review Queue — uncertain predictions (confidence 0.4–0.6)

Usage:
    pipeline = ModerationPipeline(model_path="./best_model")
    result = pipeline.predict("some comment text")
    # Returns a dict: {"decision": ..., "layer": ..., "confidence": ..., "category": ...}
"""

import re


# ---------------------------------------------------------------------------
# Layer 1 — Regex blocklist
# ---------------------------------------------------------------------------

BLOCKLIST: dict[str, list[re.Pattern]] = {
    "direct_threat": [
        # TODO: add 5+ patterns in Part 5
    ],
    "self_harm_directed": [
        # TODO: add 4+ patterns in Part 5
    ],
    "doxxing_stalking": [
        # TODO: add 4+ patterns in Part 5
    ],
    "dehumanization": [
        # TODO: add 4+ patterns in Part 5
    ],
    "coordinated_harassment": [
        # TODO: add 3+ patterns in Part 5
    ],
}


def input_filter(text: str) -> dict | None:
    """
    Returns a block decision dict if any blocklist pattern matches, else None.
    Iterates over all categories and all patterns within each category so the
    triggering category is always identifiable in the returned decision.
    """
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                }
    return None


# ---------------------------------------------------------------------------
# Layer 2 + 3 — Calibrated model & human review queue
# ---------------------------------------------------------------------------

class ModerationPipeline:
    """
    Three-layer content moderation pipeline.

    Parameters
    ----------
    model_path : str
        Path to the saved best mitigated DistilBERT model from Part 4.
    """

    BLOCK_THRESHOLD = 0.6
    ALLOW_THRESHOLD = 0.4

    def __init__(self, model_path: str):
        # TODO: load tokenizer, model, and calibrator in Part 5
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.calibrator = None

    def _get_calibrated_confidence(self, text: str) -> float:
        """
        Tokenize text, run through the model, and return a calibrated
        probability score for the toxic class.
        """
        # TODO: implement in Part 5
        raise NotImplementedError

    def predict(self, text: str) -> dict:
        """
        Run the three-layer pipeline on a single comment.

        Returns
        -------
        dict with keys:
            decision    : "block" | "allow" | "review"
            layer       : "input_filter" | "model" | "review_queue"
            confidence  : float (1.0 for input_filter decisions)
            category    : str (only present for input_filter decisions)
        """
        # --- Layer 1: input filter ---
        filter_result = input_filter(text)
        if filter_result is not None:
            return filter_result

        # --- Layer 2: calibrated model ---
        confidence = self._get_calibrated_confidence(text)

        if confidence >= self.BLOCK_THRESHOLD:
            return {"decision": "block", "layer": "model", "confidence": confidence}

        if confidence <= self.ALLOW_THRESHOLD:
            return {"decision": "allow", "layer": "model", "confidence": confidence}

        # --- Layer 3: human review queue ---
        return {"decision": "review", "layer": "review_queue", "confidence": confidence}
