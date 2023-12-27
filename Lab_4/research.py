from collections import defaultdict
import gym
import numpy as np
from evaluation import evaluate_agent
from main import train_agent
import pandas as pd
eps_list = np.linspace(0, 1, 50)
avg_steps = []
avg_losses = []
avg_draws = []
avg_wins = []
avg_reward = []
NUM_EPISODES = 10000
NUM_TRIALS = 10000
import matplotlib.pyplot as plt

def plot_results(results_df):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    fig.suptitle('Agent Performance vs. Epsilon')

    results_df['Steps'].plot(ax=axes[0, 0], title='Steps per Episode', marker='o')
    results_df['Losses'].plot(ax=axes[0, 1], title='Losses per Episode', marker='o')
    results_df['Draws'].plot(ax=axes[1, 0], title='Draws per Episode', marker='o')
    results_df['Wins'].plot(ax=axes[1, 1], title='Wins per Episode', marker='o')
    results_df['Reward'].plot(ax=axes[2, 0], title='Average Reward per Episode', marker='o')

    # Customizing the layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Assuming you have run the code to generate the results_df

def main():
    env = gym.make('Blackjack-v1', natural=False)
    for eps in eps_list:
        q_table = defaultdict(int, {})
        q_table = train_agent(q_table, env, NUM_EPISODES, eps)
        results = evaluate_agent(q_table, env, NUM_TRIALS)
        avg_steps.append(results[0])
        avg_losses.append(results[1])
        avg_draws.append(results[2])
        avg_wins.append(results[3])
        avg_reward.append(results[4])
    results_df = pd.DataFrame({'eps':eps_list,'Steps':avg_steps,'Losses':avg_losses, 'Draws':avg_draws,
                   'Wins':avg_wins, 'Reward':avg_reward })
    results_df.set_index('eps', inplace=True)
    print(results_df)
    plot_results(results_df)


if __name__ == '__main__':
    main()
