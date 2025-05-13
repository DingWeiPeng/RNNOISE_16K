# RNNOISE_16K
RNNOISE operates at 48 kHz, raising an interesting question: How does it perform at 16 kHz? This project is based on the paper:
A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement and the original implementation: xiph/rnnoise.

## Overview
RNNOISE combines digital signal processing (DSP) techniques with deep learning to deliver robust noise suppression. Its key innovation lies in using a GRU network to estimate the noise spectrum—traditionally the most challenging task—making high-quality denoising more accessible.

### Key Features
•	C-optimized preprocessing: Noise/reverberation injection and feature extraction (e.g., band energy, pitch period, pitch correlation)
•	Data-driven transient noise removal: Achieves what classical DSP methods struggle with
•	Phase preservation: The phase spectrum remains unmodified (may introduce some distortion)
•	Triangular filterbank: Used for frequency band partitioning

### How RNNOISE Works
1.	Preprocessing: Audio is augmented with noise/reverb, then processed in C to extract: 	Per-band energy, Pitch period and correlation
2.	GRU-based suppression: The network predicts gain per frequency band, interpolated to individual bins for denoising.
3.	Output: The denoised audio retains the original phase but may exhibit distortion due to spectral modifications.

#### Filterbank Design
The frequency bands are divided using a triangular filterbank (illustrated below).
