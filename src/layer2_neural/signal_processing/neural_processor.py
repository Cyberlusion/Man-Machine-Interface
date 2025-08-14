#!/usr/bin/env python3
"""
Neural Signal Processing Pipeline for Cyberlusion
Secure, real-time processing of neural signals with cryptographic integrity
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import secrets
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes


class SignalType(Enum):
    """Neural signal types"""
    EEG = "electroencephalography"
    EMG = "electromyography"
    ECoG = "electrocorticography"
    LFP = "local_field_potential"
    SPIKE = "spike_train"


class SecurityLevel(Enum):
    """Security classification for neural data"""
    PUBLIC = 0
    CONFIDENTIAL = 1
    SECRET = 2
    TOP_SECRET = 3


@dataclass
class NeuralPacket:
    """Secure neural data packet"""
    timestamp: float
    signal_type: SignalType
    data: np.ndarray
    channel_ids: List[int]
    security_level: SecurityLevel
    integrity_hash: str
    encrypted: bool = False
    
    def verify_integrity(self, key: bytes) -> bool:
        """Verify packet integrity using HMAC"""
        computed_hash = hmac.new(
            key,
            self.data.tobytes() + str(self.timestamp).encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(computed_hash, self.integrity_hash)


class SecureNeuralProcessor:
    """
    Main neural signal processing pipeline with security features
    """
    
    def __init__(self, sampling_rate: int = 30000, channels: int = 128):
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.session_key = get_random_bytes(32)
        self.packet_counter = 0
        
        # Initialize filters
        self._init_filters()
        
        # Initialize spike detector
        self.spike_detector = SpikeDetector(sampling_rate)
        
        # Initialize decoder
        self.decoder = NeuralDecoder()
        
        # Security monitoring
        self.anomaly_detector = AnomalyDetector()
        
        # Circular buffer for real-time processing
        self.buffer_size = sampling_rate * 2  # 2 seconds
        self.buffer = np.zeros((channels, self.buffer_size))
        self.buffer_idx = 0
        
    def _init_filters(self):
        """Initialize signal filters"""
        nyquist = self.sampling_rate / 2
        
        # Spike band (300-5000 Hz)
        self.spike_filter = signal.butter(
            4, [300/nyquist, 5000/nyquist], 
            btype='bandpass', output='sos'
        )
        
        # LFP band (1-300 Hz)
        self.lfp_filter = signal.butter(
            4, [1/nyquist, 300/nyquist],
            btype='bandpass', output='sos'
        )
        
        # Notch filter for line noise (50/60 Hz)
        self.notch_50 = signal.iirnotch(50/nyquist, Q=30)
        self.notch_60 = signal.iirnotch(60/nyquist, Q=30)
        
    def process_raw_signal(self, raw_data: np.ndarray) -> NeuralPacket:
        """
        Process raw neural signal with security checks
        
        Args:
            raw_data: Raw signal data (channels x samples)
            
        Returns:
            Processed and secured neural packet
        """
        # Security check: Validate input dimensions
        if raw_data.shape[0] != self.channels:
            raise ValueError(f"Invalid channel count: {raw_data.shape[0]}")
            
        # Check for anomalies (potential attacks or faults)
        if self.anomaly_detector.check(raw_data):
            self._handle_anomaly(raw_data)
            
        # Apply filters
        filtered_data = self._apply_filters(raw_data)
        
        # Extract features
        spikes = self.spike_detector.detect(filtered_data)
        lfp_features = self._extract_lfp_features(filtered_data)
        
        # Create secure packet
        packet = self._create_secure_packet(
            filtered_data, 
            SignalType.SPIKE,
            SecurityLevel.CONFIDENTIAL
        )
        
        # Update circular buffer
        self._update_buffer(filtered_data)
        
        self.packet_counter += 1
        
        return packet
        
    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply cascaded filters to neural signal"""
        # Remove line noise
        filtered = signal.filtfilt(*self.notch_50, data, axis=1)
        filtered = signal.filtfilt(*self.notch_60, filtered, axis=1)
        
        # Apply band-pass filter
        filtered = signal.sosfiltfilt(self.spike_filter, filtered, axis=1)
        
        # Adaptive noise cancellation
        filtered = self._adaptive_filter(filtered)
        
        return filtered
        
    def _adaptive_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Adaptive filter using LMS algorithm for noise cancellation
        """
        mu = 0.01  # Learning rate
        n_taps = 32
        
        filtered = np.zeros_like(data)
        
        for ch in range(self.channels):
            # Simple LMS implementation
            w = np.zeros(n_taps)
            for i in range(n_taps, data.shape[1]):
                x = data[ch, i-n_taps:i][::-1]
                y = np.dot(w, x)
                e = data[ch, i] - y
                w += mu * e * x
                filtered[ch, i] = e
                
        return filtered
        
    def _extract_lfp_features(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract local field potential features"""
        # Apply LFP filter
        lfp_data = signal.sosfiltfilt(self.lfp_filter, data, axis=1)
        
        # Compute power spectral density
        freqs, psd = signal.welch(
            lfp_data, 
            fs=self.sampling_rate,
            nperseg=1024
        )
        
        # Extract band powers
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (100, 300)
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_powers[band_name] = np.mean(psd[:, idx], axis=1)
            
        return {
            'psd': psd,
            'frequencies': freqs,
            'band_powers': band_powers,
            'phase': self._compute_phase(lfp_data)
        }
        
    def _compute_phase(self, data: np.ndarray) -> np.ndarray:
        """Compute instantaneous phase using Hilbert transform"""
        analytic = signal.hilbert(data, axis=1)
        phase = np.angle(analytic)
        return phase
        
    def _create_secure_packet(
        self, 
        data: np.ndarray,
        signal_type: SignalType,
        security_level: SecurityLevel
    ) -> NeuralPacket:
        """Create cryptographically secure data packet"""
        timestamp = self._get_secure_timestamp()
        
        # Compute integrity hash
        integrity_hash = hmac.new(
            self.session_key,
            data.tobytes() + str(timestamp).encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Encrypt if high security
        if security_level >= SecurityLevel.SECRET:
            data = self._encrypt_data(data)
            encrypted = True
        else:
            encrypted = False
            
        packet = NeuralPacket(
            timestamp=timestamp,
            signal_type=signal_type,
            data=data,
            channel_ids=list(range(self.channels)),
            security_level=security_level,
            integrity_hash=integrity_hash,
            encrypted=encrypted
        )
        
        return packet
        
    def _encrypt_data(self, data: np.ndarray) -> np.ndarray:
        """Encrypt neural data using AES-256-GCM"""
        # Flatten array for encryption
        flat_data = data.flatten()
        
        # Generate nonce
        nonce = get_random_bytes(12)
        
        # Create cipher
        cipher = AES.new(self.session_key, AES.MODE_GCM, nonce=nonce)
        
        # Encrypt
        ciphertext, tag = cipher.encrypt_and_digest(flat_data.tobytes())
        
        # Combine nonce, tag, and ciphertext
        encrypted = nonce + tag + ciphertext
        
        # Reshape to original dimensions
        return np.frombuffer(encrypted, dtype=np.uint8).reshape(data.shape[0], -1)
        
    def _get_secure_timestamp(self) -> float:
        """Get cryptographically secure timestamp"""
        import time
        # Add random jitter to prevent timing attacks
        jitter = secrets.randbits(16) / 1e6
        return time.time() + jitter
        
    def _update_buffer(self, data: np.ndarray):
        """Update circular buffer for real-time processing"""
        n_samples = data.shape[1]
        
        if self.buffer_idx + n_samples <= self.buffer_size:
            self.buffer[:, self.buffer_idx:self.buffer_idx + n_samples] = data
            self.buffer_idx += n_samples
        else:
            # Wrap around
            remaining = self.buffer_size - self.buffer_idx
            self.buffer[:, self.buffer_idx:] = data[:, :remaining]
            self.buffer[:, :n_samples - remaining] = data[:, remaining:]
            self.buffer_idx = n_samples - remaining
            
    def _handle_anomaly(self, data: np.ndarray):
        """Handle detected anomalies in neural signal"""
        # Log anomaly
        self._log_security_event("ANOMALY_DETECTED", data)
        
        # Apply additional filtering
        # Could trigger fail-safe mode if severe
        pass
        
    def _log_security_event(self, event_type: str, data: Any):
        """Log security-relevant events"""
        import json
        import datetime
        
        event = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'event_type': event_type,
            'packet_counter': self.packet_counter,
            'data_hash': hashlib.sha256(str(data).encode()).hexdigest()
        }
        
        # In production, send to secure logging system
        print(f"Security Event: {json.dumps(event)}")


class SpikeDetector:
    """
    Real-time spike detection with multiple algorithms
    """
    
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.threshold_multiplier = 4.0  # Standard deviations
        self.refractory_period = int(0.001 * sampling_rate)  # 1ms
        
    def detect(self, data: np.ndarray) -> Dict[int, List[int]]:
        """
        Detect spikes using adaptive thresholding
        
        Returns:
            Dictionary mapping channel to spike times
        """
        spikes = {}
        
        for ch in range(data.shape[0]):
            channel_data = data[ch, :]
            
            # Compute adaptive threshold
            threshold = self._compute_threshold(channel_data)
            
            # Find threshold crossings
            crossings = self._find_crossings(channel_data, threshold)
            
            # Apply refractory period
            valid_spikes = self._apply_refractory(crossings)
            
            if len(valid_spikes) > 0:
                spikes[ch] = valid_spikes
                
        return spikes
        
    def _compute_threshold(self, data: np.ndarray) -> float:
        """Compute adaptive threshold using robust MAD estimator"""
        # Median Absolute Deviation (robust to outliers/spikes)
        mad = np.median(np.abs(data - np.median(data)))
        
        # Convert to standard deviation equivalent
        std_estimate = mad * 1.4826
        
        return -self.threshold_multiplier * std_estimate
        
    def _find_crossings(self, data: np.ndarray, threshold: float) -> List[int]:
        """Find negative threshold crossings"""
        crossings = []
        
        below_threshold = data < threshold
        
        # Find edges (transitions from above to below threshold)
        edges = np.diff(below_threshold.astype(int))
        spike_starts = np.where(edges == 1)[0]
        
        for start in spike_starts:
            # Find minimum within window
            window_end = min(start + 30, len(data))  # ~1ms window
            min_idx = start + np.argmin(data[start:window_end])
            crossings.append(min_idx)
            
        return crossings
        
    def _apply_refractory(self, spikes: List[int]) -> List[int]:
        """Apply refractory period constraint"""
        if len(spikes) <= 1:
            return spikes
            
        valid_spikes = [spikes[0]]
        
        for spike in spikes[1:]:
            if spike - valid_spikes[-1] >= self.refractory_period:
                valid_spikes.append(spike)
                
        return valid_spikes


class NeuralDecoder:
    """
    Neural signal decoder for intention extraction
    """
    
    def __init__(self):
        self.calibrated = False
        self.decoder_weights = None
        self.kalman_filter = None
        
    def calibrate(self, training_data: np.ndarray, labels: np.ndarray):
        """Calibrate decoder with training data"""
        # In production, use sophisticated ML models
        # This is a simplified linear decoder
        from sklearn.linear_model import Ridge
        
        model = Ridge(alpha=1.0)
        model.fit(training_data, labels)
        
        self.decoder_weights = model.coef_
        self.calibrated = True
        
        # Initialize Kalman filter for smoothing
        self._init_kalman_filter(labels.shape[1])
        
    def decode(self, features: np.ndarray) -> np.ndarray:
        """Decode neural features to control signals"""
        if not self.calibrated:
            raise RuntimeError("Decoder not calibrated")
            
        # Linear decoding
        decoded = np.dot(features, self.decoder_weights.T)
        
        # Kalman filtering for smoothing
        if self.kalman_filter:
            decoded = self._kalman_update(decoded)
            
        # Apply safety limits
        decoded = np.clip(decoded, -1.0, 1.0)
        
        return decoded
        
    def _init_kalman_filter(self, state_dim: int):
        """Initialize Kalman filter for state estimation"""
        from filterpy.kalman import KalmanFilter
        
        kf = KalmanFilter(dim_x=state_dim, dim_z=state_dim)
        
        # State transition matrix (assuming smooth motion)
        kf.F = np.eye(state_dim)
        
        # Measurement matrix
        kf.H = np.eye(state_dim)
        
        # Process noise
        kf.Q = np.eye(state_dim) * 0.01
        
        # Measurement noise
        kf.R = np.eye(state_dim) * 0.1
        
        # Initial covariance
        kf.P = np.eye(state_dim) * 1.0
        
        self.kalman_filter = kf
        
    def _kalman_update(self, measurement: np.ndarray) -> np.ndarray:
        """Update Kalman filter with new measurement"""
        self.kalman_filter.predict()
        self.kalman_filter.update(measurement)
        return self.kalman_filter.x


class AnomalyDetector:
    """
    Detect anomalies in neural signals that could indicate
    attacks, hardware faults, or physiological issues
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.baseline = None
        self.threshold = 5.0  # Z-score threshold
        
    def check(self, data: np.ndarray) -> bool:
        """Check for anomalies in neural data"""
        # Check for saturation
        if self._check_saturation(data):
            return True
            
        # Check for excessive noise
        if self._check_noise(data):
            return True
            
        # Check for unusual patterns
        if self._check_patterns(data):
            return True
            
        return False
        
    def _check_saturation(self, data: np.ndarray) -> bool:
        """Check if channels are saturated"""
        max_val = np.iinfo(np.int16).max * 0.95
        saturated = np.abs(data) > max_val
        
        # If more than 10% samples are saturated
        if np.mean(saturated) > 0.1:
            return True
            
        return False
        
    def _check_noise(self, data: np.ndarray) -> bool:
        """Check for excessive noise levels"""
        # Compute RMS
        rms = np.sqrt(np.mean(data**2, axis=1))
        
        if self.baseline is None:
            self.baseline = rms
            return False
            
        # Z-score test
        z_scores = np.abs((rms - self.baseline) / (self.baseline + 1e-10))
        
        if np.any(z_scores > self.threshold):
            return True
            
        # Update baseline (exponential moving average)
        self.baseline = 0.95 * self.baseline + 0.05 * rms
        
        return False
        
    def _check_patterns(self, data: np.ndarray) -> bool:
        """Check for unusual signal patterns"""
        # Check for periodicity (could indicate interference)
        for ch in range(data.shape[0]):
            autocorr = np.correlate(data[ch], data[ch], mode='same')
            
            # Look for strong periodic components
            peaks, properties = signal.find_peaks(
                autocorr,
                prominence=np.std(autocorr) * 3
            )
            
            if len(peaks) > 10:  # Too many periodic peaks
                return True
                
        return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = SecureNeuralProcessor(sampling_rate=30000, channels=128)
    
    # Simulate neural data
    duration = 1.0  # seconds
    n_samples = int(duration * processor.sampling_rate)
    
    # Generate synthetic neural signal
    np.random.seed(42)
    raw_signal = np.random.randn(processor.channels, n_samples) * 50
    
    # Add some spikes
    for ch in range(10):
        spike_times = np.random.randint(0, n_samples, 20)
        for t in spike_times:
            if t > 10 and t < n_samples - 10:
                # Add spike waveform
                raw_signal[ch, t-10:t+10] -= np.hamming(20) * 200
                
    # Process signal
    packet = processor.process_raw_signal(raw_signal)
    
    # Verify packet integrity
    if packet.verify_integrity(processor.session_key):
        print("✓ Packet integrity verified")
    else:
        print("✗ Packet integrity check failed")
        
    print(f"Processed {n_samples} samples")
    print(f"Security level: {packet.security_level.name}")
    print(f"Encrypted: {packet.encrypted}")