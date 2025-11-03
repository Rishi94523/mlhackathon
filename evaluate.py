import os
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np
from typing import Dict, List

from hangman_env import HangmanEnvironment
from hmm_model import HangmanHMM
from rl_agent import HangmanRLAgent


class HangmanEvaluator:
    """
    Evaluator for the Hangman AI system on test data.
    """
    
    def __init__(self, test_file: str = 'test.txt',
                 hmm_model_path: str = 'models/hmm_model.pkl',
                 rl_model_path: str = 'models/rl_agent.pth'):
        
        self.test_file = test_file
        self.hmm_model_path = hmm_model_path
        self.rl_model_path = rl_model_path
        
        # Load test words
        self.load_test_words()
        
        # Initialize models
        self.hmm = HangmanHMM()
        self.agent = HangmanRLAgent(state_size=100)
        self.env = HangmanEnvironment(max_wrong_guesses=6)
        
        # Load trained models
        self.load_models()
    
    def load_test_words(self):
        """Load the test word list."""
        with open(self.test_file, 'r') as f:
            self.test_words = [line.strip().lower() for line in f if line.strip()]
        print(f"Loaded {len(self.test_words)} test words")
    
    def load_models(self):
        """Load the trained HMM and RL models."""
        if not os.path.exists(self.hmm_model_path):
            raise FileNotFoundError(f"HMM model not found at {self.hmm_model_path}. Please train the model first.")
        
        if not os.path.exists(self.rl_model_path):
            raise FileNotFoundError(f"RL model not found at {self.rl_model_path}. Please train the model first.")
        
        print("Loading trained models...")
        self.hmm.load(self.hmm_model_path)
        self.agent.load(self.rl_model_path)
        
        # Set agent to evaluation mode (no exploration)
        self.agent.epsilon = 0
        print("Models loaded successfully!")
    
    def play_single_game(self, word: str, verbose: bool = False) -> Dict:
        """
        Play a single game of Hangman.
        
        Args:
            word: The target word to guess
            verbose: Whether to print game progress
            
        Returns:
            Dictionary with game results
        """
        # Reset environment
        state_info = self.env.reset(word)
        masked_word, guessed_letters, lives_left, done, info = state_info
        
        if verbose:
            print(f"\nPlaying game for word: {word}")
            print(f"Initial state: {masked_word}")
        
        # Track game history
        guess_history = []
        
        steps = 0
        max_steps = 26
        
        while not done and steps < max_steps:
            # Get available actions
            available_actions = self.env.get_available_actions()
            
            if not available_actions:
                break
            
            # Get HMM predictions
            hmm_probs = self.hmm.get_letter_probabilities(masked_word, guessed_letters)
            
            # Encode state
            state = self.agent.encode_state(masked_word, guessed_letters, lives_left, hmm_probs)
            
            # Agent selects action (no exploration in evaluation)
            action_letter = self.agent.act(state, available_actions, training=False)
            
            if action_letter is None:
                break
            
            # Take action
            prev_masked = masked_word
            masked_word, guessed_letters, lives_left, done, reward, info = self.env.step(action_letter)
            
            # Record guess
            correct = action_letter in word
            guess_history.append({
                'letter': action_letter,
                'correct': correct,
                'state_before': prev_masked,
                'state_after': masked_word,
                'lives_left': lives_left
            })
            
            if verbose:
                status = "✓" if correct else "✗"
                print(f"  Guess: '{action_letter}' {status} -> {masked_word} (Lives: {lives_left})")
            
            steps += 1
        
        # Game results
        result = {
            'word': word,
            'won': info['won'],
            'wrong_guesses': info['wrong_guesses'],
            'repeated_guesses': info['repeated_guesses'],
            'total_guesses': steps,
            'final_state': masked_word,
            'guess_history': guess_history
        }
        
        if verbose:
            if info['won']:
                print(f"  Result: WIN! (Wrong guesses: {info['wrong_guesses']})")
            else:
                print(f"  Result: LOSS (Wrong guesses: {info['wrong_guesses']})")
        
        return result
    
    def evaluate_all(self, max_games: int = None, verbose: bool = False, 
                    save_results: bool = True) -> Dict:
        """
        Evaluate the AI on all test words.
        
        Args:
            max_games: Maximum number of games to play (None for all)
            verbose: Whether to print detailed progress
            save_results: Whether to save detailed results to file
            
        Returns:
            Dictionary with overall evaluation results
        """
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        # Determine number of games
        num_games = min(max_games, len(self.test_words)) if max_games else len(self.test_words)
        test_subset = self.test_words[:num_games]
        
        print(f"Evaluating on {num_games} games...")
        
        # Initialize counters
        total_wins = 0
        total_losses = 0
        total_wrong_guesses = 0
        total_repeated_guesses = 0
        game_results = []
        
        # Progress bar
        pbar = tqdm(test_subset, desc="Playing games")
        
        for word in pbar:
            result = self.play_single_game(word, verbose=False)
            game_results.append(result)
            
            # Update counters
            if result['won']:
                total_wins += 1
            else:
                total_losses += 1
            
            total_wrong_guesses += result['wrong_guesses']
            total_repeated_guesses += result['repeated_guesses']
            
            # Update progress bar
            success_rate = total_wins / (total_wins + total_losses)
            pbar.set_description(f"Success: {success_rate:.1%} | Wrong: {total_wrong_guesses} | Repeated: {total_repeated_guesses}")
        
        # Calculate final statistics
        success_rate = total_wins / num_games
        avg_wrong_guesses = total_wrong_guesses / num_games
        avg_repeated_guesses = total_repeated_guesses / num_games
        
        # Calculate final score using the provided formula
        final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
        
        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_games': num_games,
            'wins': total_wins,
            'losses': total_losses,
            'success_rate': success_rate,
            'total_wrong_guesses': total_wrong_guesses,
            'total_repeated_guesses': total_repeated_guesses,
            'avg_wrong_guesses': avg_wrong_guesses,
            'avg_repeated_guesses': avg_repeated_guesses,
            'final_score': final_score,
            'game_results': game_results if save_results else []
        }
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Games:        {num_games}")
        print(f"Wins:              {total_wins}")
        print(f"Losses:            {total_losses}")
        print(f"Success Rate:      {success_rate:.2%}")
        print(f"Total Wrong:       {total_wrong_guesses}")
        print(f"Total Repeated:    {total_repeated_guesses}")
        print(f"Avg Wrong/Game:    {avg_wrong_guesses:.2f}")
        print(f"Avg Repeated/Game: {avg_repeated_guesses:.2f}")
        print("="*60)
        print(f"FINAL SCORE: {final_score:.2f}")
        print("="*60)
        
        # Score breakdown
        print("\nScore Calculation:")
        print(f"  Success bonus:     {success_rate * 2000:.2f} (Success Rate × 2000)")
        print(f"  Wrong penalty:    -{total_wrong_guesses * 5:.2f} (Wrong Guesses × 5)")
        print(f"  Repeated penalty: -{total_repeated_guesses * 2:.2f} (Repeated Guesses × 2)")
        print(f"  Final Score:       {final_score:.2f}")
        
        # Save results if requested
        if save_results:
            results_file = f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {results_file}")
        
        # Analyze performance by word length
        if verbose:
            self.analyze_by_word_length(game_results)
        
        return results
    
    def analyze_by_word_length(self, game_results: List[Dict]):
        """Analyze performance grouped by word length."""
        print("\n" + "="*60)
        print("PERFORMANCE BY WORD LENGTH")
        print("="*60)
        
        # Group results by word length
        length_groups = {}
        for result in game_results:
            length = len(result['word'])
            if length not in length_groups:
                length_groups[length] = {'wins': 0, 'total': 0, 'wrong': 0}
            
            length_groups[length]['total'] += 1
            if result['won']:
                length_groups[length]['wins'] += 1
            length_groups[length]['wrong'] += result['wrong_guesses']
        
        # Print analysis
        print(f"{'Length':<10} {'Games':<10} {'Wins':<10} {'Success':<12} {'Avg Wrong':<10}")
        print("-" * 52)
        
        for length in sorted(length_groups.keys()):
            group = length_groups[length]
            success_rate = group['wins'] / group['total']
            avg_wrong = group['wrong'] / group['total']
            print(f"{length:<10} {group['total']:<10} {group['wins']:<10} "
                  f"{success_rate:<12.1%} {avg_wrong:<10.2f}")
    
    def test_specific_words(self, words: List[str]):
        """
        Test the AI on specific words with detailed output.
        """
        print("\n" + "="*60)
        print("TESTING SPECIFIC WORDS")
        print("="*60)
        
        for word in words:
            result = self.play_single_game(word, verbose=True)
            print()


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Hangman AI on test set')
    parser.add_argument('--test-file', type=str, default='test.txt', help='Path to test file')
    parser.add_argument('--max-games', type=int, default=None, help='Maximum number of games to play')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--test-words', nargs='+', help='Specific words to test')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HangmanEvaluator(test_file=args.test_file)
    
    # Test specific words if provided
    if args.test_words:
        evaluator.test_specific_words(args.test_words)
    
    # Run full evaluation
    results = evaluator.evaluate_all(
        max_games=args.max_games,
        verbose=args.verbose,
        save_results=True
    )
    
    # Check if we meet the 70% accuracy requirement
    if results['success_rate'] >= 0.70:
        print(f"\n✅ SUCCESS! Achieved {results['success_rate']:.2%} accuracy (>= 70% required)")
    else:
        print(f"\n⚠️  Current accuracy: {results['success_rate']:.2%} (< 70% required)")
        print("Consider adjusting hyperparameters or training for more episodes.")


if __name__ == "__main__":
    main()
