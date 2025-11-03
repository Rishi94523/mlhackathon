import numpy as np
import random
from tqdm import tqdm
import os
import json
from typing import List, Dict, Tuple

from hangman_env import HangmanEnvironment
from hmm_model import HangmanHMM
from rl_agent import HangmanRLAgent


class HangmanTrainer:
    """
    Trainer for the Hangman AI system combining HMM and RL.
    """
    
    def __init__(self, corpus_file: str = 'corpus.txt', 
                 hmm_model_path: str = 'models/hmm_model.pkl',
                 rl_model_path: str = 'models/rl_agent.pth'):
        
        self.corpus_file = corpus_file
        self.hmm_model_path = hmm_model_path
        self.rl_model_path = rl_model_path
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Load corpus
        self.load_corpus()
        
        # Initialize models
        self.hmm = HangmanHMM()
        self.agent = HangmanRLAgent(state_size=100)
        self.env = HangmanEnvironment(max_wrong_guesses=6)
    
    def load_corpus(self):
        """Load the word corpus."""
        with open(self.corpus_file, 'r') as f:
            self.words = [line.strip().lower() for line in f if line.strip()]
        print(f"Loaded {len(self.words)} words from corpus")
        
        # Split into train and validation
        random.shuffle(self.words)
        split_idx = int(len(self.words) * 0.9)
        self.train_words = self.words[:split_idx]
        self.val_words = self.words[split_idx:]
        
        print(f"Train set: {len(self.train_words)} words")
        print(f"Validation set: {len(self.val_words)} words")
    
    def train_hmm(self):
        """Train the Hidden Markov Model."""
        print("\n" + "="*50)
        print("Training Hidden Markov Model...")
        print("="*50)
        
        self.hmm.train(self.train_words)
        self.hmm.save(self.hmm_model_path)
        
        # Test HMM predictions
        print("\nTesting HMM predictions on sample words...")
        test_samples = random.sample(self.val_words, min(5, len(self.val_words)))
        
        for word in test_samples:
            masked = "_" * len(word)
            guessed = set()
            probs = self.hmm.get_letter_probabilities(masked, guessed, top_k=5)
            print(f"Word: {word} -> Top predictions: {list(probs.keys())[:5]}")
    
    def train_rl_agent(self, episodes: int = 5000, save_interval: int = 500):
        """
        Train the RL agent using the HMM predictions.
        """
        print("\n" + "="*50)
        print("Training Reinforcement Learning Agent...")
        print("="*50)
        
        # Training statistics
        episode_rewards = []
        success_rates = []
        wrong_guesses_history = []
        
        # Progress bar
        pbar = tqdm(range(episodes), desc="Training RL Agent")
        
        for episode in pbar:
            # Select random word from training set
            word = random.choice(self.train_words)
            
            # Reset environment
            state_info = self.env.reset(word)
            masked_word, guessed_letters, lives_left, done, info = state_info
            
            # Get initial HMM predictions
            hmm_probs = self.hmm.get_letter_probabilities(masked_word, guessed_letters)
            
            # Encode initial state
            state = self.agent.encode_state(masked_word, guessed_letters, lives_left, hmm_probs)
            
            episode_reward = 0
            steps = 0
            max_steps = 26  # Maximum possible guesses
            
            while not done and steps < max_steps:
                # Get available actions
                available_actions = self.env.get_available_actions()
                
                if not available_actions:
                    break
                
                # Agent selects action
                action_letter = self.agent.act(state, available_actions, training=True)
                
                if action_letter is None:
                    break
                
                action_idx = self.agent.alphabet.index(action_letter)
                
                # Take action in environment
                masked_word, guessed_letters, lives_left, done, reward, info = self.env.step(action_letter)
                
                # Update HMM predictions with new state
                hmm_probs = self.hmm.get_letter_probabilities(masked_word, guessed_letters)
                
                # Encode new state
                next_state = self.agent.encode_state(masked_word, guessed_letters, lives_left, hmm_probs)
                
                # Store experience
                self.agent.remember(state, action_idx, reward, next_state, done)
                
                # Update state
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Train on batch if enough experiences
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.replay()
            
            # Record episode statistics
            episode_rewards.append(episode_reward)
            success = info['won']
            wrong_guesses = info['wrong_guesses']
            
            # Update success rate (moving average over last 100 episodes)
            if episode % 100 == 0 and episode > 0:
                recent_successes = sum(1 for i in range(max(0, episode-100), episode) 
                                     if episode_rewards[i] > 0)
                success_rate = recent_successes / 100
                success_rates.append(success_rate)
                wrong_guesses_avg = np.mean([self.env.wrong_guesses for _ in range(100)])
                wrong_guesses_history.append(wrong_guesses_avg)
                
                pbar.set_description(f"Episode {episode} | Success Rate: {success_rate:.2%} | "
                                   f"Epsilon: {self.agent.epsilon:.4f} | "
                                   f"Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")
            
            # Update target network periodically
            if episode % 100 == 0:
                self.agent.update_target_network()
            
            # Save model periodically
            if episode % save_interval == 0 and episode > 0:
                self.agent.save(self.rl_model_path)
                self.save_training_stats(episode_rewards, success_rates, wrong_guesses_history)
        
        # Final save
        self.agent.save(self.rl_model_path)
        self.save_training_stats(episode_rewards, success_rates, wrong_guesses_history)
        
        print(f"\nTraining complete! Final success rate: {success_rates[-1]:.2%}" if success_rates else "Training complete!")
    
    def save_training_stats(self, episode_rewards: List[float], 
                           success_rates: List[float],
                           wrong_guesses: List[float]):
        """Save training statistics."""
        stats = {
            'episode_rewards': episode_rewards,
            'success_rates': success_rates,
            'wrong_guesses': wrong_guesses,
            'final_epsilon': self.agent.epsilon
        }
        
        with open('models/training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def evaluate_on_validation(self, num_games: int = 100) -> Dict:
        """
        Evaluate the trained agent on validation set.
        """
        print("\n" + "="*50)
        print("Evaluating on Validation Set...")
        print("="*50)
        
        # Set agent to evaluation mode (no exploration)
        self.agent.epsilon = 0
        
        results = {
            'total_games': num_games,
            'wins': 0,
            'losses': 0,
            'total_wrong_guesses': 0,
            'total_repeated_guesses': 0,
            'game_details': []
        }
        
        val_sample = random.sample(self.val_words, min(num_games, len(self.val_words)))
        
        for word in tqdm(val_sample, desc="Evaluating"):
            # Reset environment
            state_info = self.env.reset(word)
            masked_word, guessed_letters, lives_left, done, info = state_info
            
            # Get initial HMM predictions
            hmm_probs = self.hmm.get_letter_probabilities(masked_word, guessed_letters)
            
            # Encode initial state
            state = self.agent.encode_state(masked_word, guessed_letters, lives_left, hmm_probs)
            
            steps = 0
            max_steps = 26
            
            while not done and steps < max_steps:
                available_actions = self.env.get_available_actions()
                
                if not available_actions:
                    break
                
                # Agent selects action (no exploration)
                action_letter = self.agent.act(state, available_actions, training=False)
                
                if action_letter is None:
                    break
                
                # Take action
                masked_word, guessed_letters, lives_left, done, reward, info = self.env.step(action_letter)
                
                # Update HMM predictions
                hmm_probs = self.hmm.get_letter_probabilities(masked_word, guessed_letters)
                
                # Encode new state
                state = self.agent.encode_state(masked_word, guessed_letters, lives_left, hmm_probs)
                
                steps += 1
            
            # Record results
            if info['won']:
                results['wins'] += 1
            else:
                results['losses'] += 1
            
            results['total_wrong_guesses'] += info['wrong_guesses']
            results['total_repeated_guesses'] += info['repeated_guesses']
            
            results['game_details'].append({
                'word': word,
                'won': info['won'],
                'wrong_guesses': info['wrong_guesses'],
                'repeated_guesses': info['repeated_guesses'],
                'final_state': masked_word
            })
        
        # Calculate statistics
        results['success_rate'] = results['wins'] / results['total_games']
        results['avg_wrong_guesses'] = results['total_wrong_guesses'] / results['total_games']
        results['avg_repeated_guesses'] = results['total_repeated_guesses'] / results['total_games']
        
        # Calculate final score
        results['final_score'] = (results['success_rate'] * 2000) - \
                                 (results['total_wrong_guesses'] * 5) - \
                                 (results['total_repeated_guesses'] * 2)
        
        print(f"\nValidation Results:")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Average Wrong Guesses: {results['avg_wrong_guesses']:.2f}")
        print(f"Average Repeated Guesses: {results['avg_repeated_guesses']:.2f}")
        print(f"Final Score: {results['final_score']:.2f}")
        
        return results


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Hangman AI')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--corpus', type=str, default='corpus.txt', help='Path to corpus file')
    parser.add_argument('--skip-hmm', action='store_true', help='Skip HMM training if already trained')
    parser.add_argument('--load-models', action='store_true', help='Load existing models')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = HangmanTrainer(corpus_file=args.corpus)
    
    # Load existing models if requested
    if args.load_models:
        if os.path.exists(trainer.hmm_model_path):
            trainer.hmm.load(trainer.hmm_model_path)
            print("Loaded existing HMM model")
        
        if os.path.exists(trainer.rl_model_path):
            trainer.agent.load(trainer.rl_model_path)
            print("Loaded existing RL agent")
    else:
        # Train HMM
        if not args.skip_hmm:
            trainer.train_hmm()
        elif os.path.exists(trainer.hmm_model_path):
            trainer.hmm.load(trainer.hmm_model_path)
            print("Loaded existing HMM model")
        else:
            trainer.train_hmm()
        
        # Train RL agent
        trainer.train_rl_agent(episodes=args.episodes)
    
    # Evaluate on validation set
    trainer.evaluate_on_validation(num_games=200)


if __name__ == "__main__":
    main()
