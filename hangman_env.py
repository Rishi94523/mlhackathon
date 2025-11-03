import numpy as np
from typing import List, Tuple, Set, Optional

class HangmanEnvironment:
    """
    Hangman game environment for reinforcement learning.
    """
    
    def __init__(self, max_wrong_guesses: int = 6):
        self.max_wrong_guesses = max_wrong_guesses
        self.reset()
        
    def reset(self, word: Optional[str] = None) -> Tuple[str, Set[str], int, bool, dict]:
        """
        Reset the environment with a new word.
        
        Returns:
            Tuple of (masked_word, guessed_letters, lives_left, done, info)
        """
        self.word = word.lower() if word else ""
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.done = False
        self.won = False
        
        return self.get_state()
    
    def get_masked_word(self) -> str:
        """Get the current masked version of the word."""
        if not self.word:
            return ""
        return "".join([c if c in self.guessed_letters else "_" for c in self.word])
    
    def get_state(self) -> Tuple[str, Set[str], int, bool, dict]:
        """Get the current state of the game."""
        masked_word = self.get_masked_word()
        lives_left = self.max_wrong_guesses - self.wrong_guesses
        
        info = {
            'wrong_guesses': self.wrong_guesses,
            'repeated_guesses': self.repeated_guesses,
            'won': self.won,
            'word': self.word if self.done else None
        }
        
        return masked_word, self.guessed_letters.copy(), lives_left, self.done, info
    
    def step(self, letter: str) -> Tuple[str, Set[str], int, bool, float, dict]:
        """
        Make a guess in the game.
        
        Args:
            letter: The letter to guess
            
        Returns:
            Tuple of (masked_word, guessed_letters, lives_left, done, reward, info)
        """
        if self.done:
            return self.get_state()[:4] + (0.0, self.get_state()[4])
        
        letter = letter.lower()
        reward = 0.0
        
        # Check for repeated guess
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            reward = -0.5  # Small penalty for repeated guess
        else:
            self.guessed_letters.add(letter)
            
            if letter in self.word:
                # Correct guess
                occurrences = self.word.count(letter)
                reward = 1.0 * occurrences  # Reward based on how many letters revealed
                
                # Check if word is complete
                if "_" not in self.get_masked_word():
                    self.done = True
                    self.won = True
                    reward += 10.0  # Bonus for winning
            else:
                # Wrong guess
                self.wrong_guesses += 1
                reward = -2.0  # Penalty for wrong guess
                
                if self.wrong_guesses >= self.max_wrong_guesses:
                    self.done = True
                    self.won = False
                    reward -= 5.0  # Additional penalty for losing
        
        state = self.get_state()
        return state[:4] + (reward, state[4])
    
    def get_available_actions(self) -> List[str]:
        """Get list of letters that haven't been guessed yet."""
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        return [c for c in alphabet if c not in self.guessed_letters]
