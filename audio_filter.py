"""Audio filtering utilities.

Provides a bandpass filter (default 80 Hz – 15 kHz) to remove low-frequency
noise (hum, rumble, plosives) and high-frequency hiss from recorded WAV files
before they are muxed into the final video.

Usage:
    from audio_filter import bandpass_filter_wav

    # Filter in-place (overwrites the WAV file)
    bandpass_filter_wav("output/stream/stream_20260305-120000.wav")

    # Or with custom cutoffs
    bandpass_filter_wav("path/to/file.wav", low_hz=100, high_hz=12000, order=5)
"""
import wave
import numpy as np
from scipy.signal import butter, sosfilt


def _butter_bandpass(low: float, high: float, fs: float, order: int = 4):
    """Design a Butterworth bandpass filter (second-order sections)."""
    nyq = 0.5 * fs
    lo = max(1e-6, min(low / nyq, 1.0 - 1e-6))
    hi = max(1e-6, min(high / nyq, 1.0 - 1e-6))
    return butter(order, [lo, hi], btype='band', output='sos')


def bandpass_filter(data: np.ndarray, samplerate: int,
                    low_hz: float = 80.0, high_hz: float = 15000.0,
                    order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter to a 1-D float array.

    Parameters
    ----------
    data : np.ndarray
        Audio samples as float32/float64 in range [-1, 1].
    samplerate : int
        Sample rate in Hz.
    low_hz : float
        Lower cutoff frequency (default 80 Hz — removes rumble/hum).
    high_hz : float
        Upper cutoff frequency (default 15 000 Hz — removes hiss).
    order : int
        Filter order (default 4, i.e. 24 dB/oct rolloff per edge).

    Returns
    -------
    np.ndarray
        Filtered audio (same shape and dtype as input).
    """
    sos = _butter_bandpass(low_hz, high_hz, samplerate, order)
    return sosfilt(sos, data).astype(data.dtype)


def bandpass_filter_wav(wav_path: str, low_hz: float = 20.0,
                        high_hz: float = 16000.0, order: int = 4) -> None:
    """Read a WAV file, apply a bandpass filter, and overwrite it in-place.

    Parameters
    ----------
    wav_path : str or Path
        Path to the WAV file.
    low_hz : float
        Lower cutoff frequency in Hz (default 80).
    high_hz : float
        Upper cutoff frequency in Hz (default 15 000).
    order : int
        Butterworth filter order.
    """
    wav_path = str(wav_path)

    # --- Read ---
    with wave.open(wav_path, 'rb') as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if nframes == 0:
        return

    # Decode to float
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype='<i2').astype('float64') / 32767.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype='<i4').astype('float64') / 2147483647.0
    else:
        samples = np.frombuffer(raw, dtype='uint8').astype('float64') / 128.0 - 1.0

    # Handle multi-channel: reshape, filter each channel, then flatten
    if nchannels > 1:
        samples = samples.reshape(-1, nchannels)
        for ch in range(nchannels):
            samples[:, ch] = bandpass_filter(samples[:, ch], framerate,
                                             low_hz, high_hz, order)
        samples = samples.flatten()
    else:
        samples = bandpass_filter(samples, framerate, low_hz, high_hz, order)

    # Clip to prevent overflow
    samples = np.clip(samples, -1.0, 1.0)

    # Encode back to original sample width
    if sampwidth == 2:
        encoded = (samples * 32767.0).astype('<i2').tobytes()
    elif sampwidth == 4:
        encoded = (samples * 2147483647.0).astype('<i4').tobytes()
    else:
        encoded = ((samples + 1.0) * 128.0).astype('uint8').tobytes()

    # --- Write (overwrite) ---
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(encoded)
