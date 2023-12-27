from collections import defaultdict

import pickle
import random
import gym
from utils import select_optimal_action

alpha = 0.2
gamma = 0.8
epsilon = 0.05
NUM_EPISODES = 10000

def update(q_table, env, state, eps):
    if random.uniform(0, 1) < eps:
        action = env.action_space.sample()
    else:
        action = select_optimal_action(q_table, state, env.action_space)

    next_state, reward, done, _ = env.step(action)

    old_q_value = q_table[state][action]

    # Check if next_state has q values already
    if not q_table[next_state]:
        q_table[next_state] = {action: 0 for action in range(env.action_space.n)}

    # Calculate the new q_value
    next_max = max(q_table[next_state].values())
    new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)

    # Finally, update the q_value
    q_table[state][action] = new_q_value

    return next_state, reward, done

def train_agent(q_table, env, num_episodes, eps=0.05):
    total_steps, total_losses, total_draws, total_wins, total_reward = 0, 0, 0, 0, 0
    for i in range(num_episodes):
        state = env.reset()

        state_key = tuple(state.items())
        if not q_table[state_key]:
            q_table[state] = {action: 0 for action in range(env.action_space.n)}
        steps = 0
        done = False
        reward = 0
        while not done:
            state, reward, done = update(q_table, env, state, eps)
            steps += 1
        total_reward += reward
        if reward == -1:
            total_losses += 1
        elif reward == 0:
            total_draws += 1
        else:
            total_wins +=1

        total_steps += steps

    return q_table

def main():
    num_episodes = NUM_EPISODES
    save_path = 'q_table.pickle'
    env = gym.make('Blackjack-v1', natural=False)
    q_table = defaultdict(int, {})
    q_table = train_agent(q_table, env, num_episodes)

    with open(save_path, 'wb') as f:
        pickle.dump(dict(q_table), f)

if __name__ == '__main__':
    main()
