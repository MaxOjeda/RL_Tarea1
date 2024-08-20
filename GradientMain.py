from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent
from agents.GradientAgent import GradientBanditAgent
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

    NUM_OF_RUNS = 2000 # 2000
    NUM_OF_STEPS = 1000
    rewards = []
    optimals = []
    parameters = [(0.1, True), (0.1, False), (0.4, True), (0.4, False)]
    for i in range(4):
        results = BanditResults()
        for run_id in range(NUM_OF_RUNS):
            bandit = BanditEnv(seed=run_id, mean=4)
            num_of_arms = bandit.action_space
            agent = GradientBanditAgent(num_of_arms, alpha=parameters[i][0], baseline=parameters[i][1])  # here you might change the agent that you want to use
            best_action = bandit.best_action
            for _ in range(NUM_OF_STEPS):
                action = agent.get_action()
                reward = bandit.step(action)
                agent.learn(action, reward)
                is_best_action = action == best_action
                results.add_result(reward, is_best_action)
            results.save_current_run()

    
        avg_rewards, optimal_act_perc = show_results(results)
        rewards.append(avg_rewards)
        optimals.append(optimal_act_perc)

    for i in range(4):
        plt.plot(range(NUM_OF_STEPS), optimals[i], label=f'alpha = {parameters[i][0]}, baseline = {parameters[i][1]}') #color=['tab:orange', 'tab:blue'][i]
    plt.ylabel("% Optimal action", fontsize=15)
    plt.xlabel("Steps", fontsize=15)
    plt.legend(fontsize=13)
    plt.show()
