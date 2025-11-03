# 🧠 Bidirectional HMM + RL Concepts Explained

## Overview

This Hangman AI uses a **Bidirectional Hidden Markov Model (HMM)** combined with **Q-Learning Reinforcement Learning** to achieve high accuracy in solving Hangman puzzles.

---

## 1. 🔄 **Bidirectional HMM**

### What is it?

Traditional HMMs only look at **previous** letters to predict the next one:
```
A → P → P → ?
```
Only knows: "After APP, what comes next?"

**Bidirectional HMM** looks BOTH ways:
```
? ← P ← P ← L ← E
↓
A → P → P → ? → ?
```
Knows: "What comes between APP and LE?"

### Example in Action

**Pattern:** `_PPLE`

**Forward Context (Traditional):**
- Start of word → `?`
- Not much information!

**Backward Context (Bidirectional):**
- What letter commonly comes BEFORE 'P' in 5-letter words ending in `PLE`?
- Answer: 'A' (from APPLE, common English word)

**Result:** Bidirectional HMM confidently suggests **'A'** ✅

---

## 2. 🔗 **Trigram Models**

### Progression of N-gram Models

**Unigram:** P(letter)
```
Just frequency: E is most common (12%)
```

**Bigram:** P(letter | previous_letter)
```
"AP" → What comes after "AP"?
```

**Trigram:** P(letter | previous_2_letters)
```
"APP" → What comes after "APP"?
Much more context!
```

### Why Trigrams are Better

**Word:** APPLE

Using Bigrams:
- P(P|A) = ?
- P(P|P) = ?  (many words have PP)
- P(L|P) = ?  (many options)

Using Trigrams:
- P(P|A_) = high for APPLE
- P(P|AP) = very high! (APPLE, APPLICATION)
- P(L|PP) = very high! (APPLE, APPLICATION, APPLY)

**Result:** More accurate predictions! 🎯

---

## 3. ⚖️ **Heavy HMM Weighting (10x)**

### The Concept

In the RL agent's decision making:

```python
final_score = q_value + (hmm_probability × 10)
```

### Why Weight HMM so Heavily?

| Component | Role | Knowledge |
|-----------|------|-----------|
| **HMM** | Expert | Trained on ALL 50,000 corpus words |
| **RL Agent** | Learner | Learning from experience (20,000 episodes) |

**Early Training:**
- RL Q-values: mostly 0 (unexplored states)
- HMM probabilities: accurate from day 1
- **Result:** Agent follows HMM guidance (smart exploration)

**Late Training:**
- RL Q-values: learned from experience
- HMM probabilities: still accurate
- **Result:** RL fine-tunes HMM's suggestions

### Numeric Example

**State:** `A___E` (5-letter word)

```
Letter 'P':
- Q-value: 0.5 (RL learned this is good)
- HMM prob: 0.35 (35% chance based on corpus)
- Combined: 0.5 + (0.35 × 10) = 4.0

Letter 'G':
- Q-value: 0.2 (RL doesn't know much)
- HMM prob: 0.10 (10% chance)
- Combined: 0.2 + (0.10 × 10) = 1.2

Decision: Choose 'P' (4.0 > 1.2) ✅
```

---

## 4. 🔄 **Dynamic Prediction on Each Guess**

### Key Insight

**HMM recalculates probabilities AFTER EVERY GUESS** because the pattern changes!

### Full Game Example

**Target Word:** `APPLE`

#### **Turn 1: `_____`**
```
Pattern: Empty 5-letter word
HMM Analysis:
- Global frequency for 5-letter words
- Top predictions: E(18%), A(15%), O(12%), I(10%), S(9%)

Agent Guesses: E
Result: _____ (wrong! No 'E' in APPLE)
Lives: 5 remaining
```

#### **Turn 2: `_____` (but E is wrong)**
```
Pattern: 5-letter word, NOT containing E
HMM RECALCULATES:
- Filters out words with 'E'
- Remaining words: SHIRT, PRINT, TRUCK, SOLID...
- Top predictions: A(22%), I(18%), O(15%), T(12%)

Agent Guesses: A
Result: A____ ✅
Lives: 5 remaining
```

#### **Turn 3: `A____`**
```
Pattern: Starts with 'A', 5 letters, no 'E'
HMM RECALCULATES AGAIN:
- Matches: ABORT, ADMIN, APART, APPLY...
- Position 2-5 analysis:
  * After 'A': B(15%), P(25%), D(12%)...
- Top predictions: P(30%), B(18%), D(15%)

Agent Guesses: P
Result: APP__ ✅
Lives: 5 remaining
```

#### **Turn 4: `APP__`**
```
Pattern: APP__, 5 letters
HMM RECALCULATES:
- Vocabulary match: Only APPLE, APPLY, APPEL...
- Pattern "APP" is very specific!
- Letter frequencies in blanks:
  * Position 4: L(70%), A(15%), R(10%)
  * Position 5: Y(50%), E(40%), S(5%)
- Top predictions: L(60%), Y(25%), E(10%)

Agent Guesses: L
Result: APPL_ ✅
Lives: 5 remaining
```

#### **Turn 5: `APPL_`**
```
Pattern: APPL_, 5 letters
HMM RECALCULATES:
- Vocabulary match: APPLE, APPLY
- Last letter analysis:
  * E(50%), Y(50%)
- Remember: 'E' was guessed Turn 1 (wrong!)
- HMM knows: Either word had E or doesn't

Agent Guesses: Y (trying other option)
Result: APPL_ (wrong! It's APPLE)
Lives: 4 remaining

Turn 6: Agent guesses E again → ⚠️ REPEATED GUESS penalty!
```

---

## 5. 🎯 **Complete System Architecture**

```
┌────────────────────────────────────────────────────────────┐
│                 HANGMAN AI SYSTEM                          │
└────────────────────────────────────────────────────────────┘

INPUT: Current Game State
├─ Masked word: "A___E"
├─ Guessed letters: {e, t, a}
├─ Wrong guesses: {e, t}
└─ Lives remaining: 4

        ↓

┌────────────────────────────────────────────────────────────┐
│              HIGH-ACCURACY HMM (EXPERT)                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Vocabulary Pattern Matching                           │
│     └─ Find corpus words matching "A___E"                 │
│        Result: APPLE, AISLE, ABOVE, ABUSE, ACUTE...       │
│                                                            │
│  2. Bidirectional Context Analysis                        │
│     ├─ Forward: A → ?                                     │
│     │   Bigram P(? | A): B(12%), C(8%), P(25%)...        │
│     └─ Backward: ? ← E                                    │
│         Reverse P(? | E): L(30%), T(20%), N(18%)...       │
│                                                            │
│  3. Trigram Analysis                                      │
│     └─ If "AP_LE" → P(P|AP) = 85%                         │
│                                                            │
│  4. Position-Specific Probabilities                       │
│     └─ 5-letter words, position 2: P(20%), L(15%)...     │
│                                                            │
│  OUTPUT: P(letter | context)                              │
│  ├─ P: 0.35 (35%)                                         │
│  ├─ L: 0.25 (25%)                                         │
│  ├─ B: 0.15 (15%)                                         │
│  └─ ...                                                    │
│                                                            │
└────────────────────────────────────────────────────────────┘

        ↓  HMM Probabilities

┌────────────────────────────────────────────────────────────┐
│           Q-LEARNING AGENT (LEARNER)                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  State Representation:                                     │
│  ├─ Current pattern: "A___E"                              │
│  ├─ Guessed letters: {e,t,a}                              │
│  ├─ Lives: 4                                              │
│  └─ HMM probs: {p:0.35, l:0.25, ...}                      │
│                                                            │
│  Q-Value Lookup:                                          │
│  └─ Q["A___E:{e,t,a}:4"]["p"] = 0.8                       │
│                                                            │
│  Combined Decision:                                       │
│  ┌──────────┬────────┬────────┬────────────┐             │
│  │ Letter   │ Q-val  │ HMM×10 │ Combined   │             │
│  ├──────────┼────────┼────────┼────────────┤             │
│  │ P        │ 0.8    │ 3.5    │ 4.3 ← MAX  │             │
│  │ L        │ 0.3    │ 2.5    │ 2.8        │             │
│  │ B        │ 0.1    │ 1.5    │ 1.6        │             │
│  └──────────┴────────┴────────┴────────────┘             │
│                                                            │
│  DECISION: Guess 'P'                                      │
│                                                            │
└────────────────────────────────────────────────────────────┘

        ↓  Action

ENVIRONMENT: Execute Guess
└─ Result: "APP_E" ✅ Correct!

        ↓  Feedback

Q-LEARNING UPDATE:
└─ Q["A___E:{e,t,a}:4"]["p"] = 0.8 + α(reward + γ·max_Q_next - 0.8)
   New Q-value: 1.2 (increased because it worked!)
```

---

## 6. 🎓 **Training Process**

### Stage 1: HMM Training (Fast - 1 minute)

```python
# Process 50,000 corpus words
for word in corpus:
    # Build transition probabilities
    # Build trigram patterns  
    # Store in vocabulary
```

**Output:** Expert knowledge of English word patterns

### Stage 2: RL Training (Slow - 10 minutes for 20,000 episodes)

```python
for episode in range(20000):
    word = random.choice(corpus)
    while not done:
        # Get HMM predictions
        hmm_probs = hmm.get_probabilities(state)
        
        # Agent acts (ε-greedy with HMM guidance)
        action = agent.select_action(state, hmm_probs)
        
        # Execute and get reward
        reward, next_state = env.step(action)
        
        # Q-learning update
        agent.update(state, action, reward, next_state)
```

**Output:** Fine-tuned Q-values that work with HMM

---

## 7. 📊 **Expected Performance**

| Metric | Value |
|--------|-------|
| Training Win Rate | ~96% |
| Test Accuracy | Target: ≥70% |
| Avg Wrong Guesses | 2-3 per game |
| Repeated Guesses | Near 0 |

---

## 8. 🚀 **Why This Approach Works**

1. **HMM provides strong baseline** - 50K words of pattern knowledge
2. **Bidirectional context** - More information than traditional HMM
3. **Trigrams capture specific patterns** - "APP" → highly likely "L"
4. **Heavy HMM weighting** - Trust the expert, learn refinements
5. **Dynamic recalculation** - Adapts as word is revealed
6. **RL learns edge cases** - Handles ambiguous situations HMM misses

---

## 9. 💡 **Key Innovations**

### Innovation 1: Pattern Matching First
```python
if matching_words_from_vocabulary:
    # Count letters in blank positions directly
    # Most accurate method!
    return letter_frequencies
```

### Innovation 2: Bidirectional Scoring
```python
score = (0.3 * forward_prob +      # A → ?
         0.25 * backward_prob +      # ? → E  
         0.25 * trigram_prob +       # AP → ?
         0.2 * position_prob)        # Position 2 in 5-letter word
```

### Innovation 3: HMM-Guided Exploration
```python
if exploring:
    # Don't explore randomly!
    # Explore HMM's top 5 suggestions
    return random.choice(top_5_hmm_predictions)
```

---

## 🎯 Summary

This system combines:
- **Classical NLP** (HMM, n-grams, pattern matching)
- **Modern RL** (Q-learning, exploration strategies)
- **Domain knowledge** (English word patterns)

Result: A Hangman AI that learns like a human expert! 🏆

