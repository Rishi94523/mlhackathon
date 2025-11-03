"""
Final HMM + RL Solution for Hangman
Focused on high HMM accuracy as the foundation for RL success.
"""

import numpy as np
import random
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import pickle
from tqdm import tqdm
import os

class HighAccuracyHMM:
    """
    High-accuracy HMM using bidirectional context, trigrams, and pattern matching.
    This is the foundation - it must be extremely accurate for RL to succeed.
    """
    
    def __init__(self, smoothing=0.01):
        self.smoothing = smoothing
        
        # Core transition probabilities
        self.start_probs = Counter()
        self.bigram_probs = {}
        self.trigram_probs = {}
        
        # Position-based probabilities
        self.position_probs = {}
        
        # Reverse transitions for bidirectional context
        self.reverse_bigram_probs = {}
        
        # Global letter frequencies
        self.letter_freq = Counter()
        self.total_letters = 0
        
        # Vocabulary for pattern matching
        self.vocabulary = set()
        self.words_by_length = {}
        
    def train(self, words: List[str]):
        """Train HMM to maximum accuracy."""
        print(f"Training High-Accuracy HMM on {len(words)} words...")
        
        for word in words:
            word = word.lower().strip()
            if not word:
                continue
            
            self.vocabulary.add(word)
            
            word_len = len(word)
            if word_len not in self.words_by_length:
                self.words_by_length[word_len] = set()
            self.words_by_length[word_len].add(word)
            
            for i, letter in enumerate(word):
                # Global frequency
                self.letter_freq[letter] += 1
                self.total_letters += 1
                
                # Position-specific
                if word_len not in self.position_probs:
                    self.position_probs[word_len] = {}
                if i not in self.position_probs[word_len]:
                    self.position_probs[word_len][i] = Counter()
                self.position_probs[word_len][i][letter] += 1
                
                # Start probabilities
                if i == 0:
                    self.start_probs[letter] += 1
                
                # Bigrams (forward)
                if i > 0:
                    prev = word[i-1]
                    if prev not in self.bigram_probs:
                        self.bigram_probs[prev] = Counter()
                    self.bigram_probs[prev][letter] += 1
                    
                    # Reverse bigrams
                    if letter not in self.reverse_bigram_probs:
                        self.reverse_bigram_probs[letter] = Counter()
                    self.reverse_bigram_probs[letter][prev] += 1
                
                # Trigrams
                if i > 1:
                    trigram_key = word[i-2:i]
                    if trigram_key not in self.trigram_probs:
                        self.trigram_probs[trigram_key] = Counter()
                    self.trigram_probs[trigram_key][letter] += 1
        
        # Normalize to probabilities
        self._normalize_probabilities()
        
        print(f"HMM trained on {len(self.vocabulary)} unique words")
        print(f"Vocabulary by length: {dict(sorted([(k, len(v)) for k, v in self.words_by_length.items()]))}")
        
    def _normalize_probabilities(self):
        """Convert counts to probabilities with smoothing."""
        vocab_size = 26
        
        # Start probs
        total = sum(self.start_probs.values())
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            count = self.start_probs.get(letter, 0)
            self.start_probs[letter] = (count + self.smoothing) / (total + self.smoothing * vocab_size)
        
        # Bigram probs
        all_letters = set('abcdefghijklmnopqrstuvwxyz')
        for prev in list(self.bigram_probs.keys()):
            total = sum(self.bigram_probs[prev].values())
            normalized = Counter()
            for letter in all_letters:
                count = self.bigram_probs[prev].get(letter, 0)
                normalized[letter] = (count + self.smoothing) / (total + self.smoothing * vocab_size)
            self.bigram_probs[prev] = normalized
        
        # Reverse bigram probs
        for next_letter in list(self.reverse_bigram_probs.keys()):
            total = sum(self.reverse_bigram_probs[next_letter].values())
            normalized = Counter()
            for letter in all_letters:
                count = self.reverse_bigram_probs[next_letter].get(letter, 0)
                normalized[letter] = (count + self.smoothing) / (total + self.smoothing * vocab_size)
            self.reverse_bigram_probs[next_letter] = normalized
    
    def get_probabilities(self, masked_word: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """
        Get highly accurate probability distribution using ALL available context.
        """
        word_len = len(masked_word)
        available = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in guessed_letters]
        
        if not available:
            return {}
        
        # Strategy 1: Pattern matching with vocabulary (MOST ACCURATE)
        guessed_wrong = set(c for c in guessed_letters if c not in masked_word)
        matching_words = self._find_matching_words(masked_word, guessed_wrong)
        
        if matching_words:
            # Count letters in blank positions
            letter_counts = Counter()
            for word in matching_words:
                for i, char in enumerate(word):
                    if masked_word[i] == '_' and char in available:
                        letter_counts[char] += 1
            
            if letter_counts:
                total = sum(letter_counts.values())
                return {letter: count/total for letter, count in letter_counts.items()}
        
        # Strategy 2: Bidirectional HMM with trigrams
        letter_scores = defaultdict(float)
        
        for letter in available:
            score = 0.0
            blank_positions = [i for i, c in enumerate(masked_word) if c == '_']
            
            for pos in blank_positions:
                # Forward context (previous letters)
                prev = None
                for i in range(pos-1, -1, -1):
                    if masked_word[i] != '_':
                        prev = masked_word[i]
                        break
                
                forward_score = self.bigram_probs.get(prev, {}).get(letter, self.smoothing) if prev else self.start_probs.get(letter, self.smoothing)
                
                # Backward context (next letters)
                next_letter = None
                for i in range(pos+1, len(masked_word)):
                    if masked_word[i] != '_':
                        next_letter = masked_word[i]
                        break
                
                backward_score = self.reverse_bigram_probs.get(next_letter, {}).get(letter, self.smoothing) if next_letter else self.letter_freq.get(letter, 1) / max(self.total_letters, 1)
                
                # Trigram context
                trigram_score = 0
                if prev and pos > 1:
                    prev_prev = None
                    for i in range(pos-2, -1, -1):
                        if masked_word[i] != '_':
                            prev_prev = masked_word[i]
                            break
                    if prev_prev:
                        trigram_key = prev_prev + prev
                        if trigram_key in self.trigram_probs:
                            total_tri = sum(self.trigram_probs[trigram_key].values())
                            if total_tri > 0:
                                trigram_score = self.trigram_probs[trigram_key].get(letter, 0) / total_tri
                
                # Position-specific
                pos_score = 0
                if word_len in self.position_probs and pos in self.position_probs[word_len]:
                    total_pos = sum(self.position_probs[word_len][pos].values())
                    if total_pos > 0:
                        pos_score = self.position_probs[word_len][pos][letter] / total_pos
                
                # Combine scores
                combined = (0.3 * forward_score + 
                           0.25 * backward_score + 
                           0.25 * trigram_score + 
                           0.2 * pos_score)
                score += combined
            
            letter_scores[letter] = score / len(blank_positions) if blank_positions else score
        
        # Normalize
        total = sum(letter_scores.values())
        if total > 0:
            return {k: v/total for k, v in letter_scores.items()}
        
        # Fallback: global frequency
        return {letter: self.letter_freq[letter] / self.total_letters for letter in available}
    
    def _find_matching_words(self, pattern: str, guessed_wrong: Set[str]) -> List[str]:
        """Find vocabulary words matching the pattern."""
        word_len = len(pattern)
        candidates = self.words_by_length.get(word_len, set())
        
        matching = []
        for word in candidates:
            match = True
            for i, (p_char, w_char) in enumerate(zip(pattern, word)):
                if p_char == '_':
                    if w_char in guessed_wrong:
                        match = False
                        break
                elif p_char != w_char:
                    match = False
                    break
            
            if match:
                matching.append(word)
        
        return matching
    
    def save(self, filepath: str):
        """Save HMM model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"HMM saved to {filepath}")
    
    def load(self, filepath: str):
        """Load HMM model."""
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
        print(f"HMM loaded from {filepath}")


class QLearningAgent:
    """Q-Learning agent optimized for Hangman with HMM guidance."""
    
    def __init__(self, learning_rate=0.3, discount_factor=0.98,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.Q = {}  # Regular dict - initialize on demand
        
    def state_key(self, masked_word: str, guessed: Set[str], lives: int) -> str:
        """Create state key."""
        guessed_str = ''.join(sorted(guessed))
        return f"{masked_word}:{guessed_str}:{lives}"
    
    def select_action(self, masked_word: str, guessed: Set[str], lives: int,
                     hmm_probs: Dict[str, float], training: bool = True) -> str:
        """Select action using HMM-guided ε-greedy policy."""
        available = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in guessed]
        if not available:
            return None
        
        # Smart exploration: use HMM probabilities
        if training and random.random() < self.epsilon:
            # Choose from top HMM predictions
            sorted_letters = sorted(hmm_probs.items(), key=lambda x: x[1], reverse=True)
            top_5 = [letter for letter, _ in sorted_letters[:5] if letter in available]
            return random.choice(top_5 if top_5 else available)
        
        # Exploitation: combine Q-values with HMM (HEAVY HMM WEIGHT)
        state_key = self.state_key(masked_word, guessed, lives)
        
        best_letter = None
        best_score = float('-inf')
        
        for letter in available:
            if state_key not in self.Q:
                self.Q[state_key] = {}
            q_value = self.Q[state_key].get(letter, 0.0)
            hmm_value = hmm_probs.get(letter, 0) * 10.0  # HEAVY HMM WEIGHT
            combined = q_value + hmm_value
            
            if combined > best_score:
                best_score = combined
                best_letter = letter
        
        return best_letter if best_letter else available[0]
    
    def update(self, masked_word: str, guessed: Set[str], lives: int,
              action: str, reward: float, next_masked: str, next_guessed: Set[str],
              next_lives: int, done: bool, hmm_probs: Dict[str, float]):
        """Update Q-values."""
        state_key = self.state_key(masked_word, guessed, lives)
        next_state_key = self.state_key(next_masked, next_guessed, next_lives)
        
        if state_key not in self.Q:
            self.Q[state_key] = {}
        current_q = self.Q[state_key].get(action, 0.0)
        
        if done:
            max_next_q = 0
        else:
            available_next = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in next_guessed]
            if available_next:
                if next_state_key not in self.Q:
                    self.Q[next_state_key] = {}
                max_next_q = max([self.Q[next_state_key].get(a, 0.0) for a in available_next], default=0)
            else:
                max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.Q[state_key][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """Save agent."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Agent saved to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Load agent."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def train_hmm_rl(corpus_file='corpus.txt', episodes=20000):
    """Train HMM + RL system."""
    from hangman_env import HangmanEnvironment
    
    print("\n" + "="*60)
    print("HMM + Q-LEARNING HANGMAN SYSTEM")
    print("="*60)
    
    # Load corpus
    with open(corpus_file, 'r') as f:
        all_words = [line.strip().lower() for line in f if line.strip()]
    
    random.shuffle(all_words)
    split_idx = int(len(all_words) * 0.9)
    train_words = all_words[:split_idx]
    val_words = all_words[split_idx:]
    
    print(f"Training: {len(train_words)} words")
    print(f"Validation: {len(val_words)} words")
    
    # Train HMM
    print("\n1. Training HMM...")
    hmm = HighAccuracyHMM(smoothing=0.01)
    hmm.train(train_words)
    
    os.makedirs('models', exist_ok=True)
    hmm.save('models/hmm_model.pkl')
    
    # Train RL Agent
    print("\n2. Training Q-Learning Agent...")
    agent = QLearningAgent()
    env = HangmanEnvironment(max_wrong_guesses=6)
    
    wins = 0
    
    pbar = tqdm(range(episodes), desc="Training")
    
    for episode in pbar:
        # Sample word
        word = random.choice(train_words)
        
        state = env.reset(word)
        masked_word, guessed_letters, lives_left, done, info = state
        
        while not done:
            # Get HMM probabilities
            hmm_probs = hmm.get_probabilities(masked_word, guessed_letters)
            
            # Select action
            action = agent.select_action(masked_word, guessed_letters, lives_left, hmm_probs, training=True)
            if action is None:
                break
            
            # Take action
            new_state = env.step(action)
            next_masked, next_guessed, next_lives, done, reward, info = new_state
            
            # Update Q-values
            agent.update(masked_word, guessed_letters, lives_left, action, reward,
                        next_masked, next_guessed, next_lives, done, hmm_probs)
            
            masked_word, guessed_letters, lives_left = next_masked, next_guessed, next_lives
        
        if info['won']:
            wins += 1
        
        if episode % 100 == 0:
            win_rate = wins / (episode + 1)
            pbar.set_description(f"Win Rate: {win_rate:.1%} | ε: {agent.epsilon:.3f}")
    
    agent.save('models/rl_agent.pkl')
    
    print(f"\nTraining complete! Final win rate: {wins/episodes:.1%}")
    
    return hmm, agent


def evaluate(test_file='test.txt'):
    """Evaluate on test set."""
    from hangman_env import HangmanEnvironment
    
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Load models
    hmm = HighAccuracyHMM()
    hmm.load('models/hmm_model.pkl')
    agent = QLearningAgent.load('models/rl_agent.pkl')
    agent.epsilon = 0  # No exploration
    
    # Load test words
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    env = HangmanEnvironment(max_wrong_guesses=6)
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for word in tqdm(test_words, desc="Testing"):
        state = env.reset(word)
        masked_word, guessed_letters, lives_left, done, info = state
        
        while not done:
            hmm_probs = hmm.get_probabilities(masked_word, guessed_letters)
            action = agent.select_action(masked_word, guessed_letters, lives_left, hmm_probs, training=False)
            
            if action is None:
                break
            
            state = env.step(action)
            masked_word, guessed_letters, lives_left, done, reward, info = state
        
        if info['won']:
            wins += 1
        total_wrong += info['wrong_guesses']
        total_repeated += info['repeated_guesses']
    
    # Results
    success_rate = wins / len(test_words)
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Total Wrong: {total_wrong}")
    print(f"Final Score: {final_score:.2f}")
    print(f"{'='*60}")
    
    if success_rate >= 0.70:
        print(f"✅ {success_rate:.2%} >= 70%")
    
    return success_rate


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        evaluate()
    else:
        train_hmm_rl()
        evaluate()

