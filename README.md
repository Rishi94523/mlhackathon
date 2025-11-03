# 🎯 Hangman AI: Bidirectional HMM + Reinforcement Learning

An intelligent Hangman solver combining **Bidirectional Hidden Markov Models** with **Deep Q-Learning Reinforcement Learning** to achieve 70%+ accuracy.

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy tqdm

# Train the model
python final_hmm_rl.py

# Evaluate on test set
python final_hmm_rl.py eval
```

## 📊 Performance

- **Training Accuracy:** ~96%
- **Test Target:** ≥70%
- **Avg Wrong Guesses:** 2-3 per game
- **Training Time:** ~10 minutes (20,000 episodes)

## 🧠 Key Innovations

### 1. Bidirectional HMM
Unlike traditional HMMs that only look backward, our HMM analyzes context in BOTH directions:

```
Pattern: _PPLE
Forward:  ? → P (what comes before P?)
Backward: P → L → E (what fits this ending?)
Result: Predicts 'A' with high confidence! (APPLE)
```

### 2. Trigram Patterns
Goes beyond bigrams to capture more context:
- **Bigram:** "AP" → ?
- **Trigram:** "APP" → L (85% confidence for APPLE)

### 3. Heavy HMM Weighting (10x)
```python
final_decision = q_value + (hmm_probability × 10)
```
The RL agent heavily weights HMM predictions, treating it as an expert guide.

### 4. Dynamic Recalculation
HMM probabilities are **recalculated after every guess** as the pattern changes:
```
Turn 1: _____ → Guess 'E'
Turn 2: _____ (no E) → Recalculate! → Guess 'A'
Turn 3: A____ → Recalculate! → Guess 'P'
Turn 4: APP__ → Recalculate! → Guess 'L'
```

## 📖 Documentation

- **[CONCEPTS.md](CONCEPTS.md)** - Detailed explanation of all concepts
- **[Notebook](improved_models.ipynb)** - Interactive training example

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│   High-Accuracy HMM (Expert)    │
│  • Bidirectional context        │
│  • Trigram patterns             │
│  • Vocabulary matching          │
│  • Position-specific probs      │
└──────────┬──────────────────────┘
           │ P(letter | pattern)
           ↓
┌─────────────────────────────────┐
│  Q-Learning Agent (Learner)     │
│  • State: pattern + context     │
│  • Action: guess letter         │
│  • Reward: win/lose/correct     │
│  • Decision: Q + HMM×10         │
└──────────┬──────────────────────┘
           │
           ↓
      Best Letter Guess
```

## 📁 Files

- `final_hmm_rl.py` - Main HMM+RL implementation
- `hangman_env.py` - Game environment
- `hackman.ipynb` - Entire Hackman project submission
- `Analysis Report.pdf` - Report outlining the submission details
- `corpus.txt` - Training corpus (50,000 words)
- `test.txt` - Test set (2,000 words)

## 🎓 How It Works

1. **HMM Training**: Learns patterns from 50K words
   - Letter frequencies
   - Bigram/trigram transitions
   - Position-specific probabilities
   - Vocabulary storage for pattern matching

2. **RL Training**: Plays 20,000 games
   - Uses HMM for smart exploration
   - Learns Q-values for state-action pairs
   - Epsilon-greedy with decay
   - Immediate online updates

3. **Evaluation**: Tests on unseen words
   - No exploration (ε=0)
   - Pure exploitation
   - Combines Q-values + HMM predictions

## 🔬 Technical Details

### HMM Features
- **Smoothing:** Laplace smoothing (α=0.01)
- **Context:** Bidirectional (forward + backward)
- **N-grams:** Unigrams, bigrams, trigrams
- **Matching:** Direct vocabulary lookup

### RL Configuration
- **Algorithm:** Q-Learning
- **Learning rate:** α=0.3
- **Discount factor:** γ=0.98
- **Exploration:** ε-greedy (1.0 → 0.01)
- **Episodes:** 20,000

### Rewards
- Win: +25
- Lose: -8
- Correct guess: +2
- Wrong guess: -1.5
- Repeated guess: -1

## 📈 Results

```
Training: 96.2% accuracy
Evaluation: Testing on 2000 words...
```

## 🤝 Contributing

This is a hackathon project demonstrating the power of combining classical NLP (HMM) with modern RL.

## 📝 License

This project was made for academic/hackathon purposes.

## 👤 Author

Raihan Naeem
Prem M Thakur
Rishi DV
Noel George Jose

---

⭐ **Star this repo if you found it helpful!**
