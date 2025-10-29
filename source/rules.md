# Speech Processing Final Project - Planning Document
## Topic 1: Voiced/Unvoiced/Background Classification


---


## Apply — Applying Speech Processing Techniques


To achieve accurate **Voiced / Unvoiced / Background** classification, we will implement a **complete speech signal processing pipeline**, consisting of three main stages:


---


### **1. Pre-Processing (Tiền xử lý)**


**Goal:** Prepare raw audio for feature extraction.


**Steps:**
- **Framing:** Divide the continuous speech signal into small overlapping frames (20–30 ms).
- **Noise filtering:** Apply simple denoising (e.g., spectral gating) and amplitude normalization.
- **Windowing:** Apply a **Hamming window** to smooth frame edges and reduce spectral leakage.


**Implementation in code:**
```python
frames = librosa.util.frame(signal, frame_length=480, hop_length=160)
windowed = frames * np.hamming(480)
```


This process ensures that every short-time segment represents a quasi-stationary signal suitable for time–frequency analysis.


---


### **2. Feature Extraction (Trích xuất đặc trưng)**


**Goal:** Derive measurable acoustic features for classification.


**Features used (depending on algorithm):**


| Type | Feature | Description |
|------|----------|-------------|
| **Time-domain** | **Energy** | Frame energy (∑x²). Indicates voiced intensity. |
| | **Zero Crossing Rate (ZCR)** | Number of sign changes per frame. Distinguishes noisy vs periodic segments. |
| **Frequency-domain** | **Pitch (F0)** | Estimated fundamental frequency (present only in voiced frames). |
| | **Spectral Centroid** | Center of spectral mass, higher for noisy/unvoiced frames. |
| | **MFCCs (13–40)** | Capture vocal tract shape; used in ML and CNN models. |
| | **Spectral Flatness / Roll-off / Bandwidth** | Distinguish tonal vs noise-like spectra. |


**Example code snippet:**
```python
energy = np.sum(frame ** 2)
zcr = np.mean(librosa.feature.zero_crossing_rate(frame))
mfcc = librosa.feature.mfcc(y=frame, sr=16000, n_mfcc=13)
```


---


### **3. Classification (Phân loại)**


**Goal:** Assign each frame to one of three labels:
- **Voiced**
- **Unvoiced**
- **Background**


**Approaches used across the 4 algorithms:**


| Algorithm | Method | Decision Logic |
|------------|---------|----------------|
| **1. Time-Domain Rule-Based** | Threshold-based | Energy ↑ & ZCR ↓ → **Voiced**; Energy ↓ & ZCR ↑ → **Unvoiced**; Energy ≈ 0 → **Background** |
| **2. Frequency-Domain Spectral** | Multi-feature rules | Combine Energy + Spectral Centroid + Flatness to refine classification |
| **3. Machine Learning (SVM)** | Supervised learning | Use 40+ extracted features to train a 3-class SVM |
| **4. Deep Learning (CNN)** | End-to-end feature learning | Train CNN on Mel-spectrograms to predict class directly |


**Threshold example (Algorithm 1):**
```python
if energy < 0.01:

## 4 Algorithm Approaches (Summary)

### Algorithm 1: Time-Domain Rule-Based

**Approach:** Manual feature calculation + thresholds

**Features:**
- Energy: Sum of squared samples
- ZCR: Number of times signal crosses zero

**Classification Logic:**
```
if energy < low_threshold:
    → Background
elif zcr > high_threshold:
    → Unvoiced
else:
    → Voiced
```

**Why Pick This:**
- Baseline (simplest)
- No training needed
- Shows understanding of acoustic features
- Fast execution

**Expected Accuracy:** 75-80%

---

### Algorithm 2: Frequency-Domain Spectral

**Approach:** Frequency analysis + feature-based classification

**Features:**
- MFCC (13 coefficients)
- Spectral Centroid (frequency center of mass)
- Spectral Flatness (tonal vs noisy)
- Harmonic-to-Noise Ratio

**Method:**
- Multi-threshold decision or Gaussian Mixture Model

**Why Pick This:**
- Industry-standard features (MFCC)
- More robust than time-domain only
- Perceptually motivated

**Implementation Note:**
- Show manual MFCC calculation (educational)
- Then use librosa for efficiency (allowed)

**Expected Accuracy:** 82-87%

---

### Algorithm 3: Machine Learning (SVM)

**Approach:** Train SVM from scratch on our data

**Features (40+ dimensions):**
- Time: Energy, ZCR, Autocorrelation
- Frequency: Spectral features (centroid, spread, rolloff, flatness)
- Cepstral: MFCC (13) + Deltas (13) + Delta-Deltas (13)

**Training:**
- Extract features from labeled frames
- Normalize features
- Grid search for best hyperparameters (C, gamma)
- Train with 5-fold cross-validation

**Why Pick This:**
- Supervised learning
- Learns optimal decision boundaries
- Feature importance analysis
- Fast inference

**Implementation Note:**
- Must train from scratch (not pre-trained)
- Show training process and logs

**Expected Accuracy:** 88-92%

---

### Algorithm 4: Deep Learning (CNN)

**Approach:** Build and train CNN from random initialization

**Input:**
- Mel-spectrograms (2D images: time × frequency)

**Architecture:**
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch normalization + Dropout
- Global average pooling
- Dense classifier (3 classes)

**Training:**
- 50-100 epochs with early stopping
- Data augmentation (add noise, time stretch)
- Adam optimizer, learning rate scheduling

**Why Pick This:**
- State-of-the-art approach
- End-to-end learning (learns features automatically)
- Highest accuracy potential

**Implementation Note:**
- Must train from scratch (random weight initialization)
- Show training curves (proves we trained it)
- Save training history

**Expected Accuracy:** 90-95%

---

## Speaker Recognition Experiment

**Goal:** Identify which of 3 members is speaking

**Method:**
- Use features from V/U/B classification (MFCC, spectral features)
- Aggregate frame-level to utterance-level (mean, std statistics)
- Train 3-class classifier (Member1, Member2, Member3)

**Why This Works:**
- Voiced segments contain speaker-specific info (F0, formants)
- MFCC captures vocal tract shape (unique per person)
- Each person has unique voice "fingerprint"

**Implementation:**
- Extract features from all 120 recordings
- Train Random Forest or SVM from scratch
- Evaluate with confusion matrix

**Expected Accuracy:** 70-85%

---

## Technology Stack

**Required Libraries:**
```
numpy          - Arrays, FFT
scipy          - Signal processing
librosa        - Audio processing, MFCC
soundfile      - WAV file I/O
scikit-learn   - SVM, Random Forest (train from scratch)
tensorflow     - CNN (build and train from scratch)
matplotlib     - Visualizations
seaborn        - Statistical plots
pandas         - Data tables
jupyter        - Notebook environment
```

---

