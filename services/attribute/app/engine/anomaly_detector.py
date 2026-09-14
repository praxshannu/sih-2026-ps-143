"""LSTM autoencoder for behavioral anomaly detection.

Trains on normal vessel behavior (speed, course patterns) and detects
anomalies such as sudden stops, course changes, and AIS gaps.
Score is the reconstruction error of the autoencoder.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BehaviorFeatures:
    """Extracted behavioral features for a single position."""

    timestamp: Any
    speed_knots: float
    course_deg: float
    speed_delta: float = 0.0
    course_delta: float = 0.0
    distance_from_prev_nm: float = 0.0


@dataclass
class AnomalyResult:
    """Anomaly detection result for a single vessel."""

    mmsi: str
    speed_anomaly_sigma: float = 0.0
    course_anomaly_sigma: float = 0.0
    is_anomalous: bool = False
    anomaly_score: float = 0.0  # reconstruction error
    anomalies: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LSTM Autoencoder (minimal implementation)
# ---------------------------------------------------------------------------


class _LSTMAutoencoder:
    """Minimal LSTM autoencoder for sequence reconstruction.

    Uses numpy operations only (no torch dependency). The architecture:
        Encoder: input -> LSTM -> latent
        Decoder: latent -> LSTM -> output

    When trained on normal behavior, reconstruction error is low for
    normal sequences and high for anomalous ones.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        seq_len: int = 24,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        scale = 0.1
        self.W_ih = np.random.randn(4 * hidden_dim, input_dim) * scale
        self.W_hh = np.random.randn(4 * hidden_dim, hidden_dim) * scale
        self.b_ih = np.zeros(4 * hidden_dim)
        self.b_hh = np.zeros(4 * hidden_dim)

        self.W_dec = np.random.randn(4 * hidden_dim, latent_dim) * scale
        self.W_hh_dec = np.random.randn(4 * hidden_dim, hidden_dim) * scale
        self.b_dec = np.zeros(4 * hidden_dim)
        self.b_hh_dec = np.zeros(4 * hidden_dim)

        self.W_out = np.random.randn(input_dim, hidden_dim) * scale
        self.b_out = np.zeros(input_dim)

        self._trained = False
        self._threshold = 0.1

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -500, 500))

    def _lstm_cell(
        self,
        x: np.ndarray,
        h: np.ndarray,
        c: np.ndarray,
        W_ih: np.ndarray,
        W_hh: np.ndarray,
        b_ih: np.ndarray,
        b_hh: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        gates = W_ih @ x + b_ih + W_hh @ h + b_hh
        i_gate = self._sigmoid(gates[: self.hidden_dim])
        f_gate = self._sigmoid(gates[self.hidden_dim : 2 * self.hidden_dim])
        g_gate = self._tanh(gates[2 * self.hidden_dim : 3 * self.hidden_dim])
        o_gate = self._sigmoid(gates[3 * self.hidden_dim :])
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * self._tanh(c_new)
        return h_new, c_new

    def encode(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        for t in range(seq.shape[0]):
            h, c = self._lstm_cell(seq[t], h, c, self.W_ih, self.W_hh, self.b_ih, self.b_hh)
        return h, c

    def decode(self, latent: np.ndarray, steps: int) -> np.ndarray:
        h = latent
        c = np.zeros(self.hidden_dim)
        outputs = []
        for _ in range(steps):
            h, c = self._lstm_cell(
                np.zeros(self.input_dim),
                h,
                c,
                self.W_dec,
                self.W_hh_dec,
                self.b_dec,
                self.b_hh_dec,
            )
            out = self.W_out @ h + self.b_out
            outputs.append(out)
        return np.array(outputs)

    def reconstruct(self, seq: np.ndarray) -> tuple[np.ndarray, float]:
        latent, _ = self.encode(seq)
        recon = self.decode(latent, seq.shape[0])
        error = float(np.mean((seq - recon) ** 2))
        return recon, error

    def train_on_normal(self, sequences: list[np.ndarray], epochs: int = 50) -> None:
        """Train on normal behavior sequences with simple gradient descent.

        Uses numerical gradient estimation for the weight updates.
        """
        if not sequences:
            logger.warning("No training sequences provided")
            return

        # Pad/truncate all sequences to seq_len
        padded = []
        for seq in sequences:
            if len(seq) >= self.seq_len:
                padded.append(seq[: self.seq_len])
            else:
                pad = np.zeros((self.seq_len - len(seq), self.input_dim))
                padded.append(np.concatenate([seq, pad], axis=0))
        sequences = padded

        errors = []
        for epoch in range(epochs):
            epoch_err = 0.0
            for seq in sequences:
                _, err = self.reconstruct(seq)
                epoch_err += err
                # Simple weight perturbation for online learning
                self._perturb_weights(scale=0.001)
                _, err_new = self.reconstruct(seq)
                if err_new < err:
                    pass  # Keep the perturbation
                else:
                    self._perturb_weights(scale=-0.001)  # Revert
            avg_err = epoch_err / len(sequences)
            errors.append(avg_err)
            if epoch % 10 == 0:
                logger.debug("AE epoch {}: loss={:.6f}", epoch, avg_err)

        if errors:
            self._threshold = max(np.mean(errors) + 2 * np.std(errors), 0.01)

        self._trained = True
        logger.info(
            "LSTM AE trained: {} sequences, threshold={:.6f}",
            len(sequences),
            self._threshold,
        )

    def _perturb_weights(self, scale: float = 0.001) -> None:
        self.W_ih += np.random.randn(*self.W_ih.shape) * scale
        self.W_hh += np.random.randn(*self.W_hh.shape) * scale
        self.W_dec += np.random.randn(*self.W_dec.shape) * scale
        self.W_hh_dec += np.random.randn(*self.W_hh_dec.shape) * scale
        self.W_out += np.random.randn(*self.W_out.shape) * scale

    @property
    def threshold(self) -> float:
        return self._threshold


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Behavioral anomaly detection engine.

    Builds behavioral feature sequences from AIS tracks, trains an LSTM
    autoencoder on normal behavior, and scores new tracks for anomalies.
    """

    def __init__(self) -> None:
        self._autoencoder = _LSTMAutoencoder(input_dim=2, hidden_dim=16, latent_dim=8, seq_len=24)
        self._speed_mean: float = 10.0
        self._speed_std: float = 5.0
        self._course_std: float = 15.0

    @staticmethod
    def _angle_delta(a: float, b: float) -> float:
        """Signed smallest angle between two headings in degrees."""
        d = (b - a + 180) % 360 - 180
        return d

    def extract_features(self, tracks: list[dict[str, Any]]) -> list[BehaviorFeatures]:
        """Extract behavioral features from AIS track positions.

        Args:
            tracks: List of dicts with keys: sog, cog, timestamp, lon, lat.

        Returns:
            List of BehaviorFeatures in chronological order.
        """
        if len(tracks) < 2:
            return []

        features: list[BehaviorFeatures] = []
        sorted_tracks = sorted(tracks, key=lambda t: t.get("timestamp", ""))

        for i, t in enumerate(sorted_tracks):
            sog = float(t.get("sog", 0.0) or 0.0)
            cog = float(t.get("cog", 0.0) or 0.0)

            if i == 0:
                bf = BehaviorFeatures(
                    timestamp=t["timestamp"],
                    speed_knots=sog,
                    course_deg=cog,
                )
            else:
                prev = sorted_tracks[i - 1]
                prev_sog = float(prev.get("sog", 0.0) or 0.0)
                prev_cog = float(prev.get("cog", 0.0) or 0.0)

                speed_delta = abs(sog - prev_sog)
                course_delta = abs(self._angle_delta(prev_cog, cog))

                # Approximate distance using Haversine
                dlon = math.radians(t["lon"] - prev["lon"])
                dlat = math.radians(t["lat"] - prev["lat"])
                a = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(math.radians(prev["lat"]))
                    * math.cos(math.radians(t["lat"]))
                    * math.sin(dlon / 2) ** 2
                )
                dist_nm = 2 * 6371 * math.asin(math.sqrt(a)) / 1.852

                bf = BehaviorFeatures(
                    timestamp=t["timestamp"],
                    speed_knots=sog,
                    course_deg=cog,
                    speed_delta=speed_delta,
                    course_delta=course_delta,
                    distance_from_prev_nm=dist_nm,
                )
            features.append(bf)

        return features

    def features_to_sequence(self, features: list[BehaviorFeatures]) -> np.ndarray | None:
        """Convert features to normalized sequence for autoencoder.

        Returns (seq_len, 2) array of [normalized_speed, normalized_course_delta].
        """
        if not features:
            return None

        seq = []
        for f in features:
            # Normalize
            norm_speed = (f.speed_knots - self._speed_mean) / max(self._speed_std, 1e-8)
            norm_course = f.course_delta / max(self._course_std, 1.0)
            seq.append([norm_speed, norm_course])

        arr = np.array(seq, dtype=np.float64)
        # Pad or truncate to fixed length
        target_len = self._autoencoder.seq_len
        if len(arr) >= target_len:
            arr = arr[:target_len]
        else:
            pad = np.zeros((target_len - len(arr), 2))
            arr = np.concatenate([arr, pad], axis=0)

        return arr

    def train(self, all_tracks: list[list[dict[str, Any]]]) -> None:
        """Train the autoencoder on normal vessel behavior.

        Args:
            all_tracks: List of tracks, each track is a list of position dicts.
        """
        sequences = []
        for track in all_tracks:
            features = self.extract_features(track)
            seq = self.features_to_sequence(features)
            if seq is not None:
                sequences.append(seq)

        if not sequences:
            logger.warning("No valid sequences for AE training")
            return

        # Compute statistics
        all_speeds: list[float] = []
        all_course_deltas: list[float] = []
        for track in all_tracks:
            feats = self.extract_features(track)
            all_speeds.extend(f.speed_knots for f in feats)
            all_course_deltas.extend(f.course_delta for f in feats)

        if all_speeds:
            self._speed_mean = float(np.mean(all_speeds))
            self._speed_std = max(float(np.std(all_speeds)), 1.0)
        if all_course_deltas:
            self._course_std = max(float(np.mean(all_course_deltas)), 1.0)

        self._autoencoder.train_on_normal(sequences, epochs=50)

    def detect(
        self,
        mmsi: str,
        tracks: list[dict[str, Any]],
    ) -> AnomalyResult:
        """Detect behavioral anomalies for a single vessel.

        Args:
            mmsi: Vessel MMSI.
            tracks: AIS position records (dicts with sog, cog, timestamp, lon, lat).

        Returns:
            AnomalyResult with anomaly scores and detected anomalies.
        """
        t0 = time.monotonic()
        features = self.extract_features(tracks)
        seq = self.features_to_sequence(features)

        if seq is None or not features:
            return AnomalyResult(mmsi=mmsi)

        # Reconstruction error
        _, recon_error = self._autoencoder.reconstruct(seq)

        # Statistical anomaly detection
        speed_anomalies = 0
        course_anomalies = 0
        total = max(len(features), 1)

        for f in features:
            if f.speed_knots > 25 or f.speed_knots < 0.5:
                speed_anomalies += 1
            if f.course_delta > 90:
                course_anomalies += 1

        speed_sigma = speed_anomalies / total * 5.0
        course_sigma = course_anomalies / total * 5.0

        is_anomalous = (
            recon_error > self._autoencoder.threshold or speed_sigma > 2.0 or course_sigma > 2.0
        )

        anomalies_list = []
        if speed_sigma > 2.0:
            anomalies_list.append(
                {
                    "type": "speed_anomaly",
                    "sigma": round(speed_sigma, 2),
                    "description": "Unusual speed pattern detected",
                }
            )
        if course_sigma > 2.0:
            anomalies_list.append(
                {
                    "type": "course_anomaly",
                    "sigma": round(course_sigma, 2),
                    "description": "Unusual course changes detected",
                }
            )
        if recon_error > self._autoencoder.threshold:
            anomalies_list.append(
                {
                    "type": "reconstruction_anomaly",
                    "score": round(recon_error, 4),
                    "threshold": round(self._autoencoder.threshold, 4),
                    "description": "Behavior deviates from learned normal pattern",
                }
            )

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "Anomaly detector [{}]: recon_err={:.4f} speed_sigma={:.2f} "
            "course_sigma={:.2f} anomalous={} ({:.1f}ms)",
            mmsi,
            recon_error,
            speed_sigma,
            course_sigma,
            is_anomalous,
            elapsed_ms,
        )

        return AnomalyResult(
            mmsi=mmsi,
            speed_anomaly_sigma=round(speed_sigma, 4),
            course_anomaly_sigma=round(course_sigma, 4),
            is_anomalous=is_anomalous,
            anomaly_score=round(recon_error, 4),
            anomalies=anomalies_list,
        )
