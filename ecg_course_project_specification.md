# Course Project Specification — PTB-XL / TinyML Revision

## Title

**Design of an ECG Acquisition System with Deep-Learning-Based ECG Representation and Reconstruction**

**Course:** Information Technologies in Medicine  
**Project type:** hardware architecture + offline signal-processing/ML prototype  
**Optional extension:** quantized inference on an ESP32-S3

## 1. Objective and research question

Design an ECG acquisition architecture and train a compact **1D Convolutional Autoencoder (CAE)** that compresses fixed-length ECG signal windows into learned latent vectors and reconstructs the original windows. Evaluate waveform fidelity using **Percentage Root-mean-square Difference (PRD)** and **normalized PRD (PRDN)**. If time permits, export and quantize the model to investigate whether the **encoder** can run on an ESP32-S3.

> Research question: How does the latent-vector size affect ECG reconstruction quality and compression efficiency, and can the learned encoder be deployed on a resource-constrained microcontroller?

This is a **research/engineering demonstrator**, not a clinical ECG diagnostic device.

## 2. Dataset: PTB-XL v1.0.3

Source: https://physionet.org/content/ptb-xl/1.0.3/  
DOI: https://doi.org/10.13026/kfzx-aw45

- **21,799** ten-second, **12-lead** ECG recordings from **18,869** patients.
- Original **500 Hz** (`records500/`) and downsampled **100 Hz** (`records100/`) recordings.
- WFDB waveform data, `ptbxl_database.csv` metadata, and `scp_statements.csv` label definitions.
- Clinical **record-level** labels; do not assume beat-by-beat R-peak or PVC annotations as in MIT-BIH.
- Use official `strat_fold` splits: **1–8 training**, **9 validation**, **10 testing**. Split **before** generating overlapping windows. Do not pretrain on test records for the main held-out result.

### Initial experimental setting

- Use **Lead II only** (index/lead name `II`), matching a simple single-lead acquisition design.
- Use **100 Hz** recordings for the compact embedded-oriented prototype. Optionally repeat a small experiment at 500 Hz to inspect any lost morphology.
- Begin with a small subset of training records to validate the pipeline before scaling to the full dataset.
- Record-level diagnosis labels are **not required** to train the autoencoder. Optional analysis can compare recording-level groups such as `NORM` vs diagnostic categories, but do not assign a recording diagnosis to every individual short window as if it were a beat-level label.

## 3. Hardware concept

```text
ECG electrodes (single lead + reference/RLD)
            |
            v
Input protection / anti-aliasing
            |
            v
ADS1292R (2-channel ECG AFE, ADC, PGA, RLD)
            |
           SPI
            |
            v
ESP32-S3 (prefer a module/board with PSRAM)
            |
     USB / UART / Wi-Fi
            |
            v
PC visualization and/or latent-vector decoder
```

Design and document the hardware interface and relevant chip specifications, but **do not require assembling a patient-connected circuit** for the course project. A person-connected device requires appropriate medical electrical safety, isolation, and verification; do not connect an experimental USB/mains-grounded board directly to a person.

**Important scope consistency:** PTB-XL contains 12 leads, but the selected ADS1292R-based hardware does not acquire 12 independent leads. Lead II is the primary ML input. A full clinical 12-lead acquisition would require a more elaborate front-end, e.g., the 8-channel ADS1298 and an appropriate electrode/reference configuration, and is out of scope.

## 4. Preprocessing

1. Load each 10-second recording and extract Lead II.
2. Inspect raw ECG amplitude, noise, and baseline drift.
3. Apply a **high-pass IIR filter** for baseline-wander removal. The *cutoff* is selected **between 0.05 and 0.5 Hz**, not a band-pass of 0.05–0.5 Hz:
   - test **0.05 Hz** for better preservation of ST morphology;
   - test **0.5 Hz** for stronger baseline suppression, while acknowledging potential ST distortion;
   - use a **2nd- or 4th-order Butterworth filter in second-order-section (SOS) form** for numerical stability.
4. Optional: use a low-pass filter (e.g., around 35–40 Hz at 100-Hz sampling) only if needed, and document its effect on the waveform. The digital filter must be designed for the actual sampling rate.
5. Apply consistent amplitude scaling; preserve scaling statistics if the original physical units must be recovered.
6. **Filter the complete 10-second recording before creating windows**, not each tiny window independently. Very low high-pass cutoffs and short records can cause long filter transients or weak baseline removal.
7. Compare **offline zero-phase filtering** (`sosfiltfilt`) with **causal streaming filtering** (`sosfilt`) if planning eventual ESP32 deployment. They are not interchangeable: future samples are unavailable in real time. Define filter initialization/warm-up and document edge effects.

## 5. Windowing / sliding strides

Use fixed-length windows rather than mandatory QRS detection:

- **Window size: 256 samples** at 100 Hz (**2.56 seconds**).
- **Stride: 128 samples** (**1.28 seconds**, 50% overlap).
- Use `X.shape = [batch, 1, 256]` for the initial 1-lead model.
- Optional ablation: stride = 256 (no overlap), and another window size such as 512 samples.
- Each 10-second recording produces several related windows. All windows from one recording/patient stay entirely within its designated train, validation, or test partition.
- Overlapping windows are useful for data augmentation but are correlated; do not treat them as independent test patients.

## 6. 1D Convolutional Autoencoder

### Inputs and outputs

```text
ECG window (1 x 256)
        |
   1D CNN encoder
        |
latent vector z (8 / 16 / 32 dimensions)
        |
   1D CNN decoder
        |
reconstructed ECG window (1 x 256)
```

### Compact suggested architecture

- Encoder: `Conv1d(1,8,k=5,stride=2)` → ReLU → `Conv1d(8,16,k=5,stride=2)` → ReLU → `Conv1d(16,32,k=5,stride=2)` → projection to 8/16/32-dimensional latent vector.
- Decoder: small fully connected projection, followed by upsampling + `Conv1d` (or carefully tested transpose convolutions) back to 256 samples.
- Avoid very wide layers, attention, BatchNorm-heavy structures, or unsupported deployment operators. Confirm ONNX conversion and ESP-DL operator compatibility early.
- Loss during training: **MSE** (or a carefully motivated alternative) is fine. PRD/PRDN are **evaluation metrics**, not necessarily the training objective.
- Start with Adam, learning rate `1e-3`, batch size `64–128`, about `30–50` epochs, and early stopping.
- Train only on the training partition; tune latent dimension and preprocessing using validation data.

## 7. Reconstruction evaluation

Let `x` be a reference ECG window, `x_hat` its reconstruction, and `x_bar` the mean of the reference window.

**PRD:**

`PRD (%) = 100 * ||x - x_hat||_2 / ||x||_2`

**PRDN:**

`PRDN (%) = 100 * ||x - x_hat||_2 / ||x - x_bar||_2`

- **Lower is better.** PRDN reduces sensitivity to the original waveform's DC offset.
- Compute reconstruction metrics on the **same reference signal after consistently defined preprocessing**, and invert amplitude normalization if reporting results in original signal units.
- Guard against near-zero denominators, especially for nearly constant windows.
- Report **median, interquartile range, and mean** PRDN across test windows/recordings; show both good and poor reconstructions.
- Show original vs reconstructed traces, including P/QRS/T morphology where visible.
- Report model parameters, quantized model size (if applicable), and input/latent payload size.
- Do not claim clinical diagnostic fidelity from PRDN alone; waveform features and potential ST/QRS changes must also be inspected.

### Compression interpretation

If one 256-sample window is represented as 16-bit raw values, it contains `256 * 16 = 4096 bits` (**512 bytes**). A 16-dimensional **int8** latent vector contains `16 * 8 = 128 bits` (**16 bytes**). This gives an **idealized 32:1 payload ratio**, *excluding* metadata, scaling factors, model/decoder storage, transmission headers, and overlap. With 50% overlap, the effective ratio for a continuous stream is smaller. Do not call the system a validated compressor until the reconstruction, quantization and end-to-end bit budget are measured.

## 8. ESP32 deployment feasibility (optional)

**Recommended target:** ESP32-S3 development board with PSRAM, not a basic ESP32 as the initial target.

Deployment alternatives:

**Option A — preferred:** run **only the encoder** on the ESP32-S3, transmit a quantized latent vector to a PC, and perform reconstruction with the decoder on the PC. This tests the compression use case while reducing embedded memory and inference cost.

**Option B — harder:** run the **complete CAE** (encoder + decoder) locally on ESP32-S3. This is not required for the course project and must be demonstrated with profiling rather than assumed feasible.

Proposed workflow:

1. Train a small PyTorch 1D CAE on a desktop/laptop.
2. Export the encoder to ONNX and check compatible operations.
3. Perform **post-training INT8 quantization** using Espressif's **ESP-PPQ / ESP-DL** tools, producing a `.espdl` model.
4. Evaluate quantized outputs and reconstruction PRDN with the same held-out test signals.
5. Measure real on-board inference latency, peak RAM use, flash/model size, and energy/power if instruments are available.
6. Verify that the real-time signal filter, normalization, windowing, and quantized encoder match the Python preprocessing pipeline.

Official references:
- ESP-DL: https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html
- Quantization: https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html
- Supported operators: https://github.com/espressif/esp-dl/blob/master/operator_support_state.md

## 9. Minimal results and deliverables

### Mandatory

- Hardware block diagram and selected components, with technical specifications.
- Dataset overview and patient-wise splitting.
- Documented IIR filtering and representative signal plots.
- Sliding-window dataset with controllable stride.
- Small working 1D CAE and reconstruction examples.
- PRD and **PRDN** results for at least two latent sizes.
- Analysis of reconstruction quality and limitations.
- Source code and **15–20 page** technical report following the lecturer's assignment.

### Optional (only after mandatory work is complete)

- Comparison of different high-pass cutoff frequencies and stride settings.
- Quantization-aware or post-training quantization evaluation.
- On-board encoder inference on ESP32-S3.
- Reconstructing from encoded vectors received over serial.
- Record-level diagnostic-category analysis, if carefully separated from beat-level claims.

### Out of scope

- Training on a microcontroller.
- Custom QRS detector.
- HMM/HSMM, S4, Mamba or Transformers.
- End-to-end certified patient monitoring.
- Training a 12-lead embedded model for the first iteration.
- Clinical diagnosis claims.

## 10. Proposed repository

```text
ecg-course-project/
├── README.md
├── requirements.txt
├── data/README.md
├── notebooks/
│   ├── 01_ptbxl_exploration.ipynb
│   └── 02_reconstruction_results.ipynb
├── src/
│   ├── ptbxl.py
│   ├── filtering.py
│   ├── windowing.py
│   ├── cae.py
│   ├── train.py
│   ├── metrics.py
│   └── evaluate.py
├── embedded/              # optional
│   ├── quantize.py
│   └── esp32_s3/
├── figures/
└── report/
```

## 11. Practical milestones

1. **Data and preprocessing:** load PTB-XL Lead II, plot, filter, and verify patient-wise split.
2. **Windowing:** create fixed-size overlapping windows; test filtering and scaling.
3. **CAE:** train one small baseline model, plot reconstruction.
4. **Evaluation:** compare two latent sizes using PRDN, visual fidelity, and payload estimates.
5. **Documentation:** write ECG system architecture, chips, signal pipeline, experimental results.
6. **Optional only:** quantize encoder and test on ESP32-S3.

## 12. References

- PTB-XL v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- AHA/ACCF/HRS ECG standardization: https://doi.org/10.1016/j.jacc.2007.01.024
- PRD/PRDN example: https://pmc.ncbi.nlm.nih.gov/articles/PMC8398887/
- ADS1292R: https://www.ti.com/product/ADS1292R
- ESP-DL: https://github.com/espressif/esp-dl
