## Hackman — Hangman ML / HMM + RL experiments for ML Hackathon

Authors:
- Raihan Naeem      
- Prem M Thakur     
- Rishi D V         
- Noel George Jose

### What this project is

This repository contains a small collection of scripts and experiments for modeling and solving a Hangman-like problem using statistical and reinforcement-learning approaches. It includes data (a small corpus), a gym-like environment, HMM/RL experiments, and a few demo scripts/notebooks, built as part of the ML Hackathon for Semester 5 (AIML)

### Files of interest
- `hangman_env.py` — a Gym-like Hangman environment used by the experiments.
- `final_hmm_rl.py` — main experiment implementing HMM and reinforcement-learning approaches (final candidate).
- `demo.py` — simple demo script showing how to run parts of the project or example usage.
- `hackman.ipynb` — notebook with exploratory code, visualisations or step-by-step experiments.
- `corpus.txt` — word corpus used by environment/experiments.
- `requirements.txt` — Python dependencies needed to run the project.
- `test.txt` — small file used in testing or demos.

### Quickstart (Windows / PowerShell)
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the demo script to see a simple example:

```powershell
python demo.py
```

4. Run the main experiment/notebook (notebook can be opened with Jupyter):

```powershell
jupyter notebook hackman.ipynb
```

Or run the final script directly:

```powershell
python final_hmm_rl.py # to train
```
```powershell
python final_hmm_rl.py eval # to evaluate
```


### Notes and assumptions
- The code targets Python 3.8+ (use the interpreter referenced in your `requirements.txt`).
- The repository contains small, experimental code — expect exploratory scripts and a notebook rather than a packaged library.
- If you add or change dependencies, update `requirements.txt`.

### Contact / author
This repo was created as part of a machine-learning hackathon — see repository for author and commit history.

---
