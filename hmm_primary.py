"""
HMM-Primary Hangman Solver
Uses HMM as the main decision maker (RL just for tie-breaking)
"""

import numpy as np
import random
from collections import defaultdict, Counter
from typing import Dict, List, Set
import pickle
from tqdm import tqdm
import os

from final_hmm_rl import HighAccuracyHMM
from hangman_env import HangmanEnvironment


def evaluate_hmm_primary(test_file='test.txt'):
    """
    Evaluate using HMM as primary decision maker.
    RL is only for edge cases.
    """
    print("\n" + "="*60)
    print("HMM-PRIMARY EVALUATION")
    print("="*60)
    
    # Load HMM
    hmm = HighAccuracyHMM()
    if not os.path.exists('models/hmm_model.pkl'):
        print("Training HMM first...")
        with open('corpus.txt', 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        hmm.train(words)
        hmm.save('models/hmm_model.pkl')
    else:
        hmm.load('models/hmm_model.pkl')
    
    # Load test words
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    env = HangmanEnvironment(max_wrong_guesses=6)
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for word in tqdm(test_words[:2000], desc="Testing"):
        state = env.reset(word)
        masked_word, guessed_letters, lives_left, done, info = state
        
        while not done:
            # Get HMM probabilities
            hmm_probs = hmm.get_probabilities(masked_word, guessed_letters)
            
            if not hmm_probs:
                break
            
            # SIMPLE DECISION: Choose letter with highest HMM probability
            action = max(hmm_probs.items(), key=lambda x: x[1])[0]
            
            if action is None:
                break
            
            state = env.step(action)
            masked_word, guessed_letters, lives_left, done, reward, info = state
        
        if info['won']:
            wins += 1
        total_wrong += info['wrong_guesses']
        total_repeated += info['repeated_guesses']
    
    # Results
    num_games = min(2000, len(test_words))
    success_rate = wins / num_games
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    print(f"\n{'='*60}")
    print("RESULTS (HMM-PRIMARY)")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Total Wrong: {total_wrong}")
    print(f"Total Repeated: {total_repeated}")
    print(f"Final Score: {final_score:.2f}")
    print(f"{'='*60}")
    
    if success_rate >= 0.70:
        print(f"✅ {success_rate:.2%} >= 70%")
    else:
        print(f"Current: {success_rate:.2%} - Need {0.70 - success_rate:.1%} more")
    
    return success_rate, final_score


if __name__ == "__main__":
    evaluate_hmm_primary()

