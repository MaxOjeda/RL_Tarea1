from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent
from agents.IncrementalAgent import IncrementalAgent
import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('ggplot')

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
    epsilons = [0, 0.01, 0.1]
    for epsilon in epsilons:
        results = BanditResults()
        for run_id in range(NUM_OF_RUNS):
            bandit = BanditEnv(seed=run_id)
            num_of_arms = bandit.action_space
            agent = IncrementalAgent(num_of_arms, epsilon=epsilon)  # here you might change the agent that you want to use
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

    fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
    for i, epsilon in enumerate(epsilons):
        axs[0].plot(range(NUM_OF_STEPS), rewards[i], label=f'epsilon = {epsilon}', color=['tab:orange', 'tab:green', 'tab:blue'][i])
    axs[0].set_ylabel("Average Reward", fontsize=15)
    axs[0].set_xlabel("Steps", fontsize=15)
    axs[0].legend()

    for i, epsilon in enumerate(epsilons):
        axs[1].plot(range(NUM_OF_STEPS), optimals[i], label=f'epsilon = {epsilon}', color=['tab:orange', 'tab:green', 'tab:blue'][i])
    axs[1].set_ylabel("% Optimal action", fontsize=15)
    yticks = axs[1].get_yticks()
    axs[1].set_yticks(yticks)
    axs[1].set_yticklabels(['{:,.0f}%'.format(x*100) for x in yticks])
    axs[1].set_xlabel("Steps", fontsize=15)
    axs[1].legend()
    plt.show()
