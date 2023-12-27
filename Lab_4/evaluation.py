import pickle
import gym

from utils import select_optimal_action
NUM_EPISODES = 1000  # Adjust as needed

def evaluate_agent(q_table, env, num_trials):
    total_steps, total_losses, total_draws, total_wins, total_reward = 0, 0, 0, 0, 0
    print('Running episodes...')
    for _ in range(num_trials):
        state = env.reset()
        steps, num_losses, num_draws, num_wins = 0, 0, 0, 0
        done = False
        reward = 0
        while not done:
            next_action = select_optimal_action(q_table, state, env.action_space)
            state, reward, done, _ = env.step(next_action)
            steps += 1

        if reward == -1:  # In Blackjack-v0, -1 represents a penalty (losing a round)
            num_losses += 1
        elif reward == 0:
            num_draws += 1
        else:
            num_wins += 1

        total_losses += num_losses
        total_draws += num_draws
        total_wins += num_wins
        total_reward += reward
        total_steps += steps

    average_steps = total_steps / float(num_trials)
    average_losses = total_losses / float(num_trials)
    average_draws = total_draws / float(num_trials)
    average_wins = total_wins / float(num_trials)
    average_reward = total_reward / float(num_trials)

    return average_steps, average_losses, average_draws, average_wins, average_reward

def main():
    num_episodes = NUM_EPISODES
    q_path = 'q_table.pickle'
    env = gym.make('Blackjack-v1', natural=False)
    with open(q_path, 'rb') as f:
        q_table = pickle.load(f)
    evaluate_agent(q_table, env, num_episodes)

if __name__ == '__main__':
    main()
