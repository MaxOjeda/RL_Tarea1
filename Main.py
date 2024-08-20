from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent
from agents.IncrementalAgent import IncrementalAgent
import matplotlib.pyplot as plt
import seaborn as sns

def show_results(bandit_results: type(BanditResults)) -> None:
    print("\nAverage results")
    print("Step\tReward\tOptimal action (%)")
    average_rewards = bandit_results.get_average_rewards()
    optimal_action_percentage = bandit_results.get_optimal_action_percentage()
    for step in range(NUM_OF_STEPS):
        print(f"{step+1}\t{average_rewards[step]:0.3f}\t{optimal_action_percentage[step]:0.3f}")

    return average_rewards, optimal_action_percentage


if __name__ == "__main__":

    NUM_OF_RUNS = 500 # 2000
    NUM_OF_STEPS = 1000

    results = BanditResults()
    for run_id in range(NUM_OF_RUNS):
        bandit = BanditEnv(seed=run_id)
        num_of_arms = bandit.action_space
        agent = IncrementalAgent(num_of_arms)  # here you might change the agent that you want to use
        best_action = bandit.best_action
        for _ in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            results.add_result(reward, is_best_action)
        results.save_current_run()

    
    avg_rewards, optimal_act_perc = show_results(results)
    fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
    axs[0].plot(range(NUM_OF_STEPS), avg_rewards)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Steps")

    axs[1].plot(range(NUM_OF_STEPS), optimal_act_perc)
    axs[1].set_ylabel("% Optimal action")
    axs[1].set_xlabel("Steps")
    plt.show()
