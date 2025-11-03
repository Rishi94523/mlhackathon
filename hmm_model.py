import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import pickle

class HangmanHMM:
    """
    Hidden Markov Model for predicting letter probabilities in Hangman.
    
    The HMM models:
    - Hidden states: Letter positions in words (relative positions)
    - Emissions: Observed letters
    - Transitions: How positions flow in words
    """
    
    def __init__(self):
        self.letter_freq = defaultdict(int)  # Overall letter frequencies
        self.positional_freq = {}  # Letter frequencies by position and word length
        self.bigram_freq = defaultdict(lambda: defaultdict(int))  # Letter bigrams
        self.trigram_freq = defaultdict(lambda: defaultdict(int))  # Letter trigrams
        self.word_length_dist = defaultdict(int)  # Word length distribution
        self.pattern_freq = defaultdict(lambda: defaultdict(int))  # Pattern-based frequencies
        self.vocabulary = set()
        
    def train(self, words: List[str]):
        """
        Train the HMM on a corpus of words.
        """
        print(f"Training HMM on {len(words)} words...")
        
        for word in words:
            word = word.lower().strip()
            if not word:
                continue
                
            self.vocabulary.add(word)
            word_len = len(word)
            self.word_length_dist[word_len] += 1
            
            # Update letter frequencies
            for i, char in enumerate(word):
                self.letter_freq[char] += 1
                
                # Positional frequencies
                key = (word_len, i)
                if key not in self.positional_freq:
                    self.positional_freq[key] = defaultdict(int)
                self.positional_freq[key][char] += 1
                
                # Relative position (beginning, middle, end)
                rel_pos = self._get_relative_position(i, word_len)
                rel_key = (word_len, rel_pos)
                if rel_key not in self.positional_freq:
                    self.positional_freq[rel_key] = defaultdict(int)
                self.positional_freq[rel_key][char] += 1
            
            # Update bigrams and trigrams
            for i in range(len(word) - 1):
                self.bigram_freq[word[i]][word[i+1]] += 1
            
            for i in range(len(word) - 2):
                trigram_key = word[i:i+2]
                self.trigram_freq[trigram_key][word[i+2]] += 1
        
        print("HMM training complete!")
    
    def _get_relative_position(self, index: int, word_len: int) -> str:
        """Get relative position category."""
        if word_len <= 2:
            return 'all'
        elif index == 0:
            return 'start'
        elif index == word_len - 1:
            return 'end'
        elif index < word_len / 3:
            return 'early'
        elif index > 2 * word_len / 3:
            return 'late'
        else:
            return 'middle'
    
    def get_letter_probabilities(self, masked_word: str, guessed_letters: set, 
                                top_k: int = 26) -> Dict[str, float]:
        """
        Get probability distribution over unguessed letters given current game state.
        
        Args:
            masked_word: Current state like "H_LL_"
            guessed_letters: Set of already guessed letters
            top_k: Number of top letters to return
            
        Returns:
            Dictionary mapping letters to probabilities
        """
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        available = [c for c in alphabet if c not in guessed_letters]
        
        if not available:
            return {}
        
        word_len = len(masked_word)
        letter_scores = defaultdict(float)
        
        # 1. Base frequency scores
        total_freq = sum(self.letter_freq.values())
        for letter in available:
            if total_freq > 0:
                letter_scores[letter] = self.letter_freq[letter] / total_freq * 0.2
        
        # 2. Positional probability
        blank_positions = [i for i, c in enumerate(masked_word) if c == '_']
        for pos in blank_positions:
            # Exact position
            key = (word_len, pos)
            if key in self.positional_freq:
                pos_total = sum(self.positional_freq[key].values())
                for letter in available:
                    if letter in self.positional_freq[key] and pos_total > 0:
                        letter_scores[letter] += (self.positional_freq[key][letter] / pos_total) * 0.3
            
            # Relative position
            rel_pos = self._get_relative_position(pos, word_len)
            rel_key = (word_len, rel_pos)
            if rel_key in self.positional_freq:
                rel_total = sum(self.positional_freq[rel_key].values())
                for letter in available:
                    if letter in self.positional_freq[rel_key] and rel_total > 0:
                        letter_scores[letter] += (self.positional_freq[rel_key][letter] / rel_total) * 0.2
        
        # 3. Bigram and trigram probabilities
        revealed_letters = [(i, c) for i, c in enumerate(masked_word) if c != '_']
        
        for pos, char in revealed_letters:
            # Check bigrams
            if pos > 0 and masked_word[pos-1] == '_':
                bigram_total = sum(self.bigram_freq[char].values())
                for letter in available:
                    if letter in self.bigram_freq and char in self.bigram_freq[letter]:
                        if bigram_total > 0:
                            score = self.bigram_freq[letter][char] / bigram_total
                            letter_scores[letter] += score * 0.15
            
            if pos < word_len - 1 and masked_word[pos+1] == '_':
                if char in self.bigram_freq:
                    bigram_total = sum(self.bigram_freq[char].values())
                    for letter in available:
                        if letter in self.bigram_freq[char] and bigram_total > 0:
                            score = self.bigram_freq[char][letter] / bigram_total
                            letter_scores[letter] += score * 0.15
        
        # 4. Pattern matching with vocabulary
        pattern = masked_word.lower()
        matching_words = self._find_matching_words(pattern, guessed_letters)
        if matching_words:
            letter_counts = Counter()
            for word in matching_words:
                for i, char in enumerate(word):
                    if pattern[i] == '_' and char in available:
                        letter_counts[char] += 1
            
            if letter_counts:
                total_count = sum(letter_counts.values())
                for letter, count in letter_counts.items():
                    letter_scores[letter] += (count / total_count) * 0.4
        
        # Normalize scores to probabilities
        total_score = sum(letter_scores.values())
        if total_score > 0:
            probabilities = {letter: score/total_score for letter, score in letter_scores.items()}
        else:
            # Fallback to uniform distribution
            probabilities = {letter: 1.0/len(available) for letter in available}
        
        # Return top k letters
        sorted_letters = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_letters[:top_k])
    
    def _find_matching_words(self, pattern: str, guessed_letters: set) -> List[str]:
        """Find words from vocabulary that match the current pattern."""
        matching = []
        pattern_len = len(pattern)
        
        for word in self.vocabulary:
            if len(word) != pattern_len:
                continue
            
            match = True
            for i, (p_char, w_char) in enumerate(zip(pattern, word)):
                if p_char == '_':
                    # This position should not have been guessed
                    if w_char in guessed_letters:
                        match = False
                        break
                elif p_char != w_char:
                    match = False
                    break
            
            if match:
                matching.append(word)
        
        return matching
    
    def save(self, filepath: str):
        """Save the trained model."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'letter_freq': dict(self.letter_freq),
                'positional_freq': dict(self.positional_freq),
                'bigram_freq': dict(self.bigram_freq),
                'trigram_freq': dict(self.trigram_freq),
                'word_length_dist': dict(self.word_length_dist),
                'pattern_freq': dict(self.pattern_freq),
                'vocabulary': self.vocabulary
            }, f)
        print(f"HMM model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.letter_freq = defaultdict(int, data['letter_freq'])
            self.positional_freq = data['positional_freq']
            self.bigram_freq = defaultdict(lambda: defaultdict(int), data['bigram_freq'])
            self.trigram_freq = defaultdict(lambda: defaultdict(int), data['trigram_freq'])
            self.word_length_dist = defaultdict(int, data['word_length_dist'])
            self.pattern_freq = defaultdict(lambda: defaultdict(int), data['pattern_freq'])
            self.vocabulary = data['vocabulary']
        print(f"HMM model loaded from {filepath}")
