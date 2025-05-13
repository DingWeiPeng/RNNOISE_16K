# RNNOISE_16K
RNNOISE operates at 48 kHz, raising an interesting question: How does it perform at 16 kHz? This project is based on the paper:
A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement and the original implementation: xiph/rnnoise.

## Overview
RNNOISE combines digital signal processing (DSP) techniques with deep learning to deliver robust noise suppression. Its key innovation lies in using a GRU network to estimate the noise spectrum—traditionally the most challenging task—making high-quality denoising more accessible.

### Key Features of RNNOISE
* C-optimized preprocessing: Noise/reverberation injection and feature extraction (e.g., band energy, pitch period, pitch correlation)
* Data-driven transient noise removal: Achieves what classical DSP methods struggle with
* Phase preservation: The phase spectrum remains unmodified (may introduce some distortion)
* Triangular filterbank: Used for frequency band partitioning

### How RNNOISE Works
1.	Preprocessing: Audio is augmented with noise/reverb, then processed in C to extract: 	Per-band energy, Pitch period and correlation
2.	GRU-based suppression: The network predicts gain per frequency band, interpolated to individual bins for denoising.
3.	Output: The denoised audio retains the original phase but may exhibit distortion due to spectral modifications.
4.  The frequency bands are divided using a triangular filterbank (illustrated below).
<img src="https://github.com/user-attachments/assets/24662348-aa96-4117-a25c-f3686bb64a32" alt="triangular filterbank" width="900" align="center">

Based on RNNOISE and the technical analysis at https://zhuanlan.zhihu.com/p/397288851, the feature extraction framework of RNNOISE is structured as follows:
<img src="https://github.com/user-attachments/assets/2fe8979e-d45d-458e-a91e-26eb2c5f7612" alt="feature extraction" width="900" align="center">

## Work at 16000 sample rate
To adapt RNNOISE for 16kHz operation, simply modify the feature extraction framework for the 16kHz sampling rate while keeping the frame length unchanged at 10ms. At 16kHz sampling rate, each 10ms frame contains 160 samples, requiring corresponding adjustments to the spectral layout. This means the triangular filters' maximum frequency must be limited to 157 bins.

The implementation requires:
* Audio resampling to 16kHz using FFmpeg
* Running the following command to update FFT/window parameters: gcc dump_rnnoise_tables.c kiss_fft.o -I ../include -lm -o dump_rnnoise_tables

Code modifications in src/denoise.h:
* After adjusting NB_BANDS and eband20ms parameters, the feature extraction pipeline continues functioning normally.
<img src="https://github.com/user-attachments/assets/67eb0056-4f6b-4cf5-be15-f95c1f09987e" alt="change in denoise.h" width="900" align="center">

In src/denoise.c, the modifications are as follows:
<img src="https://github.com/user-attachments/assets/29fdc078-af00-402b-a860-e070ee223644" alt="change in denoise.h" width="900" align="center">

In dump_features.c, you can reduce RIR_FFT_SIZE to half of its original value. The complete compilation steps for the project are:
% ./configure
% make

After training the model, you can run:
% ./examples/rnnoise_demo <noisy_speech.wav> <output_denoised.pcm>

The complete training procedure is as follows (you may choose count=10000, meaning extracting features 10,000 times):
Step 1: Convert audio, noise, and RIR files from 48kHz to 16kHz sampling rate.

Step 2: Feature extraction and data preprocessing. Since feature extraction is time-consuming, you may choose count=10000 (extracting features 10,000 times):
% ./dump_features -rir_list rir_list.txt speech.pcm background_noise.pcm foreground_noise.pcm features.f32 
where is the number of sequences to process. The number of sequences should be at least 10,000, but more is better (200,000 or more is recommended).

Step 3: Model training. In train_rnnoise.py, the input features for the RNNoise model are determined by NB_BANDS:
python3 train_rnnoise.py features.f32 output_directory

Step 4: Compile the model into executable files:
% python3 dump_rnnoise_weights.py --quantize rnnoise_50.pth rnnoise_c
This will generate rnnoise_data.c and rnnoise_data.h files in the rnnoise_c directory.
Copy these files to src/ and rebuild RNNoise using the instructions above.

Step 5: Validate model performance. You can use shell scripts for batch noise removal and batch conversion of PCM files to WAV format:
% ./examples/rnnoise_demo <noisy_speech.wav> <output_denoised.pcm>

## Note
The model performance of this project is still being optimized. Therefore, please don't be surprised if RNNOISE performs poorly at 16kHz. I suspect the degraded noise reduction results may be caused by improper feature extraction or frequency band partitioning.

## Appendix: Useful ffmpeg Commands and Other Commands
### Generate rnnoise tables
gcc dump_rnnoise_tables.c kiss_fft.o -I ../include -lm -o dump_rnnoise_tables

### Audio format conversion commands
ffmpeg -i background_noise_v2_16k.wav -f s16le -acodec pcm_s16le background_noise_v2_16k.pcm
ffmpeg -i background_noise_v2.wav -ar 16000 -acodec pcm_s16le background_noise_v2_16k.wav  
ffmpeg -i foreground_noise_v3.wav -ar 16000 -acodec pcm_s16le foreground_noise_v3_16k.wav
ffmpeg -f f32le -ar 48000 -ac 1 -i tts_speech_48k.pcm -ar 16000 -ac 1 -f f32le tts_speech_16k.pcm
