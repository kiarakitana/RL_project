import multiprocessing as mp
from env import DataCenterEnv
import numpy as np
import argparse
from tabularQN import TabularQAgent
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import signal
import sys
from tqdm import tqdm

# Global flag for handling early termination
early_stop = False

def signal_handler(signum, frame):
    """Handle Ctrl+C by setting early_stop flag"""
    global early_stop
    print('\nEarly termination requested. Completing current batch...')
    early_stop = True

def run_episode(args):
    """Run a single episode in its own process"""
    train_path, agent_params, episode_num = args
    
    # Create environment and agent for this process
    env = DataCenterEnv(train_path)
    agent = TabularQAgent(**agent_params)
    
    state = env.observation()
    total_reward = 0
    episode_transitions = []
    terminated = False
    
    try:
        while not terminated:
            action = agent.get_action(state)
            next_state, reward, terminated = env.step(action)
            
            episode_transitions.append((
                agent.discretize_state(state),
                agent.discretize_action(action),
                reward,
                agent.discretize_state(next_state)
            ))
            
            total_reward += reward
            state = next_state
            
        return episode_num, episode_transitions, total_reward
        
    except IndexError:
        if total_reward != 0 and episode_transitions:
            return episode_num, episode_transitions, total_reward
        return episode_num, None, None

def parallel_train_tabular_q(train_path, val_path=None, num_episodes=1000, num_processes=None, batch_size=10):
    """Train tabular Q-learning agent using multiple processes with progress updates"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Set up signal handler for early termination
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Starting parallel training using {num_processes} processes...")
    
    agent = TabularQAgent()
    train_rewards_history = []
    
    agent_params = {
        'learning_rate': agent.lr,
        'discount_factor': agent.gamma,
        'epsilon_init': agent.epsilon,
        'epsilon_decay': agent.epsilon_decay,
        'epsilon_min': agent.epsilon_min
    }
    
    start_time = time.time()
    completed_episodes = 0
    
    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            while completed_episodes < num_episodes and not early_stop:
                # Prepare batch of episodes
                batch_episodes = min(batch_size, num_episodes - completed_episodes)
                episode_args = [
                    (train_path, agent_params, completed_episodes + i) 
                    for i in range(batch_episodes)
                ]
                
                # Submit batch of episodes
                futures = [executor.submit(run_episode, args) for args in episode_args]
                
                # Process results as they complete
                batch_rewards = []
                for future in as_completed(futures):
                    episode_num, transitions, reward = future.result()
                    if transitions is not None:
                        # Update Q-table with episode transitions
                        for state, action, reward, next_state in transitions:
                            agent.update_q_table(state, action, reward, next_state)
                        batch_rewards.append(reward)
                
                # Update training history and progress
                train_rewards_history.extend(batch_rewards)
                completed_episodes += len(batch_rewards)
                
                # Print progress update
                avg_reward = np.mean(batch_rewards) if batch_rewards else 0
                elapsed_time = time.time() - start_time
                print(f"\rEpisode {completed_episodes}/{num_episodes} "
                      f"| Avg Batch Reward: {avg_reward:.2f} "
                      f"| Running Time: {elapsed_time:.1f}s "
                      f"| ε: {agent.epsilon:.3f}", end="")
                
                # Decay epsilon after each batch
                agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
                
                sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    training_time = time.time() - start_time
    print(f"\n\nTraining completed in {training_time:.2f} seconds")
    print(f"Completed {completed_episodes} episodes")
    print(f"Final average reward: {np.mean(train_rewards_history[-100:]):.2f}")
    
    # Validation phase
    if val_path and not early_stop:
        print("\nStarting validation...")
        val_rewards = []
        
        for val_episode in range(10):
            val_env = DataCenterEnv(val_path)
            state = val_env.observation()
            total_reward = 0
            terminated = False
            
            try:
                while not terminated:
                    action = agent.get_action(state, validation=True)
                    next_state, reward, terminated = val_env.step(action)
                    total_reward += reward
                    state = next_state
                
                val_rewards.append(total_reward)
                print(f"Validation Episode {val_episode + 1}, Reward: {total_reward:.2f}")
                
            except IndexError:
                if total_reward != 0:
                    val_rewards.append(total_reward)
                continue
        
        if val_rewards:
            print(f"\nAverage Validation Reward: {np.mean(val_rewards):.2f}")
        return agent, train_rewards_history, val_rewards
    
    return agent, train_rewards_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='train.xlsx')
    parser.add_argument('--val_path', type=str, default='validate.xlsx')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--processes', type=int, default=None,
                      help='Number of processes to use (default: number of CPU cores)')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Number of episodes to run in parallel per batch')
    args = parser.parse_args()
    
    try:
        results = parallel_train_tabular_q(
            args.train_path, 
            args.val_path, 
            args.episodes,
            args.processes,
            args.batch_size
        )
        
        if results:
            if len(results) == 3:
                agent, train_rewards, val_rewards = results
            else:
                agent, train_rewards = results
            
            agent.save('q_table.npy')
            print("\nSaved Q-table to q_table.npy")
            
    except KeyboardInterrupt:
        print("\nTraining terminated by user. Exiting...")
        sys.exit(0)