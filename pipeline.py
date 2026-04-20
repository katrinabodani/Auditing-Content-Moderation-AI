"""
pipeline.py
-----------
Production-grade three-layer content moderation guardrail pipeline.

Layers:
    1. Input Filter: fast regex pre-filter for obvious violations (5 categories, 20+ patterns)
    2. Calibrated Model: DistilBERT with isotonic probability calibration
    3. Human Review Queue: uncertain predictions (confidence 0.4–0.6)

Usage:
    pipeline = ModerationPipeline(model_path="./checkpoints/best_model")
    result = pipeline.predict("some comment text")
    # Returns: {"decision": "block"|"allow"|"review", "layer": ..., "confidence": ..., "category": ...}
"""

import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin


# ===========================================================================
# Layer 1 — Regex Blocklist
# ===========================================================================

BLOCKLIST: dict[str, list[re.Pattern]] = {

    # ── Category 1: Direct threats of violence (5 patterns) ─────────────────
    "direct_threat": [
        # "I will/gonna/going to [kill|murder|shoot|stab|hurt] you"
        re.compile(
            r"\bi\s*(?:will|'ll|am going to|gonna|wanna)\s+(?:kill|murder|shoot|stab|hurt|destroy|end)\s+\b",
            re.IGNORECASE
        ),
        # "you are going to die / you will die"
        re.compile(
            r"\byou(?:'re|\s+are|\s+will\s+be|\s+'ll\s+be)?\s+(?:going\s+to\s+)?die\b",
            re.IGNORECASE
        ),
        # "someone should [kill|shoot|murder|stab] you"
        re.compile(
            r"\bsomeone\s+should\s+(?:kill|shoot|murder|stab|hurt)\s+\byou\b",
            re.IGNORECASE
        ),
        # "I'll find where you live / I know where you live" (threat variant)
        re.compile(
            r"\bi(?:'ll|\s+will|\s+am\s+going\s+to|\s+gonna)\s+find\s+(?:where\s+you\s+live|your\s+address)\b",
            re.IGNORECASE
        ),
        # "watch your back / you better watch out — threat framing"
        re.compile(
            r"\b(?:watch\s+your\s+back|you(?:'d)?\s+better\s+(?:watch\s+out|run|be\s+scared)|you\s+won't\s+survive)\b",
            re.IGNORECASE
        ),
    ],

    # ── Category 2: Calls for self-harm or suicide (4 patterns) ─────────────
    "self_harm_directed": [
        # "you should kill yourself"
        re.compile(
            r"\byou\s+should\s+(?:just\s+)?(?:kill|hang|hurt)\s+yourself\b",
            re.IGNORECASE
        ),
        # "go kill/hang yourself"
        re.compile(
            r"\bgo\s+(?:kill|hang|hurt)\s+yourself\b",
            re.IGNORECASE
        ),
        # "nobody would miss you if you died / no one cares if you die"
        re.compile(
            r"\b(?:nobody|no\s+one)\s+(?:would\s+miss|cares?\s+(?:about|if))\s+you\b",
            re.IGNORECASE
        ),
        # "do everyone a favour and disappear/die"
        re.compile(
            r"\bdo\s+everyone\s+a\s+favou?r\s+and\s+(?:disappear|die|end\s+it|kill\s+yourself)\b",
            re.IGNORECASE
        ),
    ],

    # ── Category 3: Doxxing and stalking threats (4 patterns) ───────────────
    "doxxing_stalking": [
        # "I know where you live"
        re.compile(
            r"\bi\s+know\s+where\s+you\s+(?:live|work|go\s+to\s+school|are)\b",
            re.IGNORECASE
        ),
        # "I will/I'll post your address / real name / info"
        re.compile(
            r"\bi(?:'ll|\s+will|\s+am\s+going\s+to|\s+gonna)\s+(?:post|share|leak|expose|reveal)\s+your\s+(?:address|location|real\s+name|info|information|details|number|phone)\b",
            re.IGNORECASE
        ),
        # "I found your real name / address / info"
        re.compile(
            r"\bi\s+(?:found|have|got)\s+your\s+(?:real\s+name|address|location|info|information|number|ip)\b",
            re.IGNORECASE
        ),
        # "everyone will know who you really are"
        re.compile(
            r"\beveryone\s+will\s+(?:know|find\s+out)\s+(?:who|where|what)\s+you\s+really\s+(?:are|live|work)\b",
            re.IGNORECASE
        ),
    ],

    # ── Category 4: Severe dehumanization (4 patterns) ──────────────────────
    "dehumanization": [
        # "[group] are not human / are subhuman"
        re.compile(
            r"\b\w+\s+are\s+(?:not\s+)?(?:sub)?human\b",
            re.IGNORECASE
        ),
        # "[group] are animals / vermin / parasites / insects"
        re.compile(
            r"\b\w+\s+are\s+(?:animals?|vermin|parasites?|insects?|rodents?|pests?|cockroaches?)\b",
            re.IGNORECASE
        ),
        # "[group] should be exterminated / eliminated / wiped out"
        re.compile(
            r"\b\w+\s+should\s+be\s+(?:exterminated|eliminated|wiped\s+out|eradicated|cleansed|removed)\b",
            re.IGNORECASE
        ),
        # "[group] are a disease / cancer / plague / virus"
        re.compile(
            r"\b\w+\s+are\s+(?:a\s+)?(?:disease|cancer|plague|virus|infection|infestation|blight)\b",
            re.IGNORECASE
        ),
    ],

    # ── Category 5: Coordinated harassment signals (3 patterns) ─────────────
    "coordinated_harassment": [
        # "everyone report [username/this/them]"
        re.compile(
            r"\beveryone\s+(?:please\s+)?report\s+(?:this|them|him|her|@?\w+)\b",
            re.IGNORECASE
        ),
        # "let's all go after / attack / target [someone]"
        re.compile(
            r"\blet'?s\s+(?:all\s+)?(?:go\s+after|attack|target|mass\s+report|brigade|dogpile)\b",
            re.IGNORECASE
        ),
        # lookahead: "raid" followed by anything (raid their profile / raid the channel etc.)
        re.compile(
            r"\braid\b(?=.{0,50}(?:profile|channel|server|account|stream|page|group))",
            re.IGNORECASE
        ),
    ],
}


def input_filter(text: str) -> dict | None:
    """
    Run the text against all blocklist patterns.

    Returns a block decision dict if any pattern matches, else None.
    Iterates over all categories so the triggering category is always
    identifiable in the returned decision — making the filter auditable.

    Parameters
    ----------
    text : str
        Raw comment text.

    Returns
    -------
    dict | None
        {"decision": "block", "layer": "input_filter",
         "category": <str>, "confidence": 1.0}
        or None if no pattern matched.
    """
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision":   "block",
                    "layer":      "input_filter",
                    "category":   category,
                    "confidence": 1.0,
                }
    return None


# ===========================================================================
# Layer 2 + 3 — Calibrated Model & Human Review Queue
# ===========================================================================

class _DistilBertWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_path: str, max_length: int = 128):
        self.model_path = model_path
        self.max_length = max_length

    def fit(self, X, y):
        self.fitted_ = True    # ← add this line
        return self

    def predict_proba(self, X):
        p = np.array(X).flatten()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.4).astype(int)


class ModerationPipeline:
    """
    Three-layer content moderation pipeline.

    Layer 1 : regex input filter (instant, no model call)
    Layer 2 : calibrated DistilBERT (confident decisions)
    Layer 3 : human review queue (uncertain decisions)

    Parameters
    ----------
    model_path : str
        Path to the saved best mitigated DistilBERT model directory.
    max_length : int
        Tokenizer max sequence length (default 128, matches training).
    """

    BLOCK_THRESHOLD  = 0.6
    ALLOW_THRESHOLD  = 0.4
    MAX_LENGTH       = 128
    DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer  = AutoTokenizer.from_pretrained(model_path)
        self.model      = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.DEVICE)
        self.model.eval()
        self.calibrator = None  # fitted in fit_calibrator()

    def fit_calibrator(self, texts: list[str], true_labels: list[int]) -> None:
        from sklearn.isotonic import IsotonicRegression
        
        print("Computing raw probabilities for calibration set...")
        raw_probs = self._raw_probs(texts)
        
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_probs, np.array(true_labels))
        print("Calibrator fitted successfully.")

    def _raw_probs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Run batched inference and return raw softmax toxic-class probabilities."""
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc   = self.tokenizer(
                    batch, max_length=self.MAX_LENGTH,
                    truncation=True, padding=True, return_tensors="pt"
                )
                enc    = {k: v.to(self.DEVICE) for k, v in enc.items()}
                logits = self.model(**enc).logits
                probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)

    def _get_calibrated_confidence(self, text: str) -> float:
        if self.calibrator is None:
            raise RuntimeError("Call fit_calibrator() before predict().")
        
        raw = self._raw_probs([text])[0]
        cal = self.calibrator.predict([raw])[0]
        return float(np.clip(cal, 0.0, 1.0))

    def predict(self, text: str) -> dict:
        """
        Run the three-layer pipeline on a single comment.

        Parameters
        ----------
        text : str
            Raw comment text.

        Returns
        -------
        dict with keys:
            decision   : "block" | "allow" | "review"
            layer      : "input_filter" | "model" | "review_queue"
            confidence : float (1.0 for input_filter decisions)
            category   : str (only present for input_filter decisions)
        """
        # Layer 1: input filter 
        filter_result = input_filter(text)
        if filter_result is not None:
            return filter_result

        # Layer 2: calibrated model
        confidence = self._get_calibrated_confidence(text)

        if confidence >= self.BLOCK_THRESHOLD:
            return {
                "decision":   "block",
                "layer":      "model",
                "confidence": confidence
            }

        if confidence <= self.ALLOW_THRESHOLD:
            return {
                "decision":   "allow",
                "layer":      "model",
                "confidence": confidence
            }

        # Layer 3: human review queue 
        return {
            "decision":   "review",
            "layer":      "review_queue",
            "confidence": confidence
        }
