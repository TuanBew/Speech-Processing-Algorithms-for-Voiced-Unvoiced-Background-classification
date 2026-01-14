# Voiced / Unvoiced / Background Classifier

This project implements three signal-processing heuristics to label speech frames as
background (-1), unvoiced (0), or voiced (1). A speaker-recognition task is used as
an indirect proxy to compare segmentation quality across algorithms.

## Problem Statement
Frame-level voice activity and voicing need to be labeled for short speech clips
without using learned VAD models. We compare three handcrafted algorithms that
assign each frame to background (-1), unvoiced (0), or voiced (1). Because there
is no frame-level ground truth, we use speaker recognition accuracy (on features
derived from the predicted labels) as a proxy for segmentation quality.

## Data
- Location: `./data/<speaker>/*.wav` (train) and `./test/<speaker>/*.wav` (test)
- Speakers: LeNguyenTuanAnh, LuongBinhMinh, TranQuangThai
- Words (10 takes each, indices 0–9): yes (/jes/), yeah (/jeə/), present (/ˈpreznt/), here (/hɪə/)
- Split sizes: 120 train files, 120 test files
- Audio: 16 kHz, mono, single channel

## Outputs
- Frame-level labels are written to `./output/<algorithm>/<speaker>/<file>_labels.tsv`
- Columns: time_s, label (-1=background, 0=unvoiced, 1=voiced)
- Algorithms: HHTC (time-domain), SSC (spectral shape), HDGC (dual-geometry)

## Algorithms
- HHTC: short-time energy/RMS gate, zero-crossing rate, autocorrelation peak,
  plus temporal smoothing.
- SSC: spectral centroid, flatness, rolloff, ZCR with percentile thresholds on
  speech-only frames.
- HDGC: log-STFT, dyadic frequency tree, EMD-like distances, two-step clustering
  (silence vs speech, then voiced vs unvoiced by high/low band ratio).

## Notebook
- Primary workflow: `Presentation.ipynb`
  - Sets paths relative to the repo root (no Colab mount required).
  - Computes features, runs algorithms, exports labels, trains SVMs for speaker
    recognition as a proxy metric, and provides visualization helpers.

## Environment
- Python packages: numpy, librosa, matplotlib, scipy, scikit-learn
- Suggested install: `pip install -r requirements.txt` (or install the above
  packages manually). A requirements file is not included by default; generate
  one if needed with `pip freeze > requirements.txt` after setting up your env.

## Running locally
1) Ensure folders exist: `data/`, `test/`, `output/` (the last will be created).
2) Open `Presentation.ipynb` and run cells in order. Key cells:
   - Path setup (defines DATA_ROOT/TEST_ROOT/OUTPUT_ROOT and checks existence).
   - Exporting algorithm outputs (writes TSVs under `output/`).
   - Speaker-ID comparison (trains SVMs per algorithm and prints accuracy and
     reports).
3) Inspect results:
   - TSV label files under `output/` per algorithm and speaker.
   - Plots and sequence stats from notebook visualization cells.

## Results (proxy metric)
- Speaker-ID accuracy is reported per algorithm on the provided train/test split;
  higher accuracy suggests cleaner V/UV/B segmentation on this dataset. Exact
  numbers depend on the local run; they print in the SVM comparison cell.

## Notes
- The project favors simplicity: no learned VAD models; all three algorithms are
  handcrafted. The SVM is only for downstream comparison.
- Audio is assumed to be correctly sampled at 16 kHz mono; resampling occurs in
  the notebook via librosa.load(sr=16000).
