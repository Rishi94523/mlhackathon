"""
Interactive Hangman Demo with HMM+RL
Shows the model playing Hangman step-by-step with predictions.
"""

import time
import random
from final_hmm_rl import HighAccuracyHMM, QLearningAgent
from hangman_env import HangmanEnvironment

def print_header():
    """Print demo header."""
    print("\n" + "="*70)
    print("🎯 HANGMAN AI DEMO - HMM + RL in Action")
    print("="*70)
    print("Watching the model play Hangman with 95% training accuracy!")
    print("="*70 + "\n")

def print_game_state(word, masked_word, guessed_letters, lives_left, hmm_probs, action=None, q_value=None):
    """Print current game state."""
    print("\n" + "-"*70)
    print(f"📝 Current State:")
    print(f"   Pattern: {masked_word}")
    print(f"   Word Length: {len(word)} letters")
    print(f"   Guessed Letters: {', '.join(sorted(guessed_letters)) if guessed_letters else 'None'}")
    print(f"   Lives Remaining: {'❤️ ' * lives_left}")
    print(f"   Progress: {sum(1 for c in masked_word if c != '_')}/{len(word)} letters revealed")
    
    if hmm_probs:
        print(f"\n   🔮 HMM Top 5 Predictions:")
        sorted_probs = sorted(hmm_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (letter, prob) in enumerate(sorted_probs, 1):
            bar = "█" * int(prob * 20)
            print(f"      {i}. '{letter}' → {prob:.1%} {bar}")
    
    if action:
        if q_value is not None:
            hmm_prob = hmm_probs.get(action, 0) if hmm_probs else 0
            combined = q_value + (hmm_prob * 50)
            print(f"\n   🤖 RL Decision:")
            print(f"      Q-value for '{action}': {q_value:.3f}")
            print(f"      HMM prob × 50: {hmm_prob * 50:.3f}")
            print(f"      Combined score: {combined:.3f}")
            print(f"\n   ✨ MODEL GUESSES: '{action.upper()}'")
        else:
            print(f"\n   ✨ MODEL GUESSES: '{action.upper()}'")

def demo_game(word, hmm, agent, delay=2):
    """Play a single game with demo output."""
    env = HangmanEnvironment(max_wrong_guesses=6)
    state = env.reset(word)
    masked_word, guessed_letters, lives_left, done, info = state
    
    print(f"\n🎲 NEW GAME STARTED!")
    print(f"   Target Word: {'*' * len(word)} (hidden)")
    print(f"   You'll see it revealed as the game progresses...")
    
    turn = 0
    
    while not done and turn < 26:
        turn += 1
        print(f"\n{'='*70}")
        print(f"TURN {turn}")
        print(f"{'='*70}")
        
        # Get HMM predictions
        hmm_probs = hmm.get_probabilities(masked_word, guessed_letters)
        
        # Get Q-value for top HMM prediction
        state_key = agent.state_key(masked_word, guessed_letters, lives_left)
        if state_key in agent.Q:
            available = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in guessed_letters]
            if hmm_probs:
                top_hmm_letter = max(hmm_probs.items(), key=lambda x: x[1])[0]
                if top_hmm_letter in available:
                    q_val = agent.Q[state_key].get(top_hmm_letter, 0.0)
                else:
                    q_val = 0.0
            else:
                q_val = 0.0
        else:
            q_val = 0.0
        
        # Agent selects action
        action = agent.select_action(masked_word, guessed_letters, lives_left, hmm_probs, training=False)
        
        # Print game state
        print_game_state(word, masked_word, guessed_letters, lives_left, hmm_probs, action, q_val)
        
        # Wait before showing result
        print(f"\n   ⏳ Processing...")
        time.sleep(delay)
        
        # Execute action
        new_state = env.step(action)
        new_masked, new_guessed, new_lives, new_done, reward, new_info = new_state
        
        # Show result
        print(f"\n   📊 RESULT:")
        if action in word:
            print(f"      ✅ CORRECT! '{action.upper()}' is in the word!")
            print(f"      New Pattern: {new_masked}")
        else:
            print(f"      ❌ WRONG! '{action.upper()}' is not in the word.")
            print(f"      Lives lost. Remaining: {new_lives}")
        
        # Update state
        masked_word, guessed_letters, lives_left, done = new_masked, new_guessed, new_lives, new_done
        
        if done:
            print(f"\n   {'='*70}")
            if new_info['won']:
                print(f"   🎉 GAME WON! The word was: {word.upper()}")
                print(f"   ✅ Turns taken: {turn}")
                print(f"   ✅ Wrong guesses: {new_info['wrong_guesses']}")
                print(f"   ✅ Final Score: {reward:.1f} points")
            else:
                print(f"   💀 GAME LOST! The word was: {word.upper()}")
                print(f"   ❌ Wrong guesses: {new_info['wrong_guesses']}")
                print(f"   ❌ Final Score: {reward:.1f} points")
            print(f"   {'='*70}")
    
    return new_info['won'], new_info['wrong_guesses'], turn


def main():
    """Main demo function."""
    print_header()
    
    # Load models
    print("📦 Loading models...")
    try:
        hmm = HighAccuracyHMM()
        hmm.load('models/hmm_model.pkl')
        print("   ✅ HMM loaded")
        
        agent = QLearningAgent.load('models/rl_agent.pkl')
        agent.epsilon = 0  # No exploration for demo
        print("   ✅ RL Agent loaded")
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        print("   Please train models first by running: python final_hmm_rl.py")
        return
    
    # Load training words from corpus.txt (where we have 95% accuracy)
    print("\n📚 Loading random words from corpus.txt...")
    try:
        with open('corpus.txt', 'r') as f:
            all_words = [line.strip().lower() for line in f if line.strip()]
        print(f"   ✅ Loaded {len(all_words)} words from corpus.txt")
        
        # Shuffle for true randomness
        random.shuffle(all_words)
        train_words = all_words  # Use all words for demo
    except FileNotFoundError:
        print(f"   ❌ Error: corpus.txt not found!")
        print(f"   Please make sure corpus.txt is in the current directory.")
        return
    except Exception as e:
        print(f"   ❌ Error loading corpus: {e}")
        return
    
    print("\n" + "="*70)
    print("🎮 DEMO MODE")
    print("="*70)
    print("Playing games with 2-second delays to show HMM+RL decisions...")
    print("Press Ctrl+C to stop at any time")
    print("="*70)
    
    # Demo stats
    games_won = 0
    games_played = 0
    total_wrong = 0
    
    try:
        while True:
            # Pick random word from training set
            word = random.choice(train_words)
            games_played += 1
            
            print(f"\n\n{'#'*70}")
            print(f"GAME #{games_played}")
            print(f"{'#'*70}")
            
            # Play game
            won, wrong, turns = demo_game(word, hmm, agent, delay=2)
            
            if won:
                games_won += 1
            total_wrong += wrong
            
            # Stats
            success_rate = games_won / games_played
            avg_wrong = total_wrong / games_played
            
            print(f"\n📈 OVERALL STATS (so far):")
            print(f"   Games Played: {games_played}")
            print(f"   Games Won: {games_won}")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Avg Wrong Guesses: {avg_wrong:.2f}")
            
            print(f"\n⏳ Waiting 3 seconds before next game...")
            print("   (Press Ctrl+C to stop)")
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 DEMO STOPPED BY USER")
        print("="*70)
        if games_played > 0:
            success_rate = games_won / games_played
            avg_wrong = total_wrong / games_played
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   Total Games: {games_played}")
            print(f"   Games Won: {games_won}")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Avg Wrong Guesses: {avg_wrong:.2f}")
        else:
            print(f"\n📊 No games completed yet.")
        print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'single':
        # Single game demo
        print_header()
        
        # Load models
        print("📦 Loading models...")
        hmm = HighAccuracyHMM()
        hmm.load('models/hmm_model.pkl')
        agent = QLearningAgent.load('models/rl_agent.pkl')
        agent.epsilon = 0
        
        # Pick a word or use provided
        if len(sys.argv) > 2:
            word = sys.argv[2].lower()
            print(f"   Using provided word: {word.upper()}")
        else:
            print(f"   Picking random word from corpus.txt...")
            with open('corpus.txt', 'r') as f:
                all_words = [line.strip().lower() for line in f if line.strip()]
            word = random.choice(all_words)
            print(f"   ✅ Random word selected from {len(all_words)} words in corpus.txt")
        
        print(f"\n🎲 Playing single game with word: {word.upper()}")
        demo_game(word, hmm, agent, delay=2)
    else:
        # Continuous demo
        main()

