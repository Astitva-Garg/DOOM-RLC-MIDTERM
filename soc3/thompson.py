import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt
from scipy.stats import beta


class ThompsonSamplingAgent(Agent):
    # Add fields
    reward_memory: np.ndarray  # A per arm value of how much reward was gathered
    count_memory: np.ndarray  # An array of the number of times an arm is pulled

    def __init__(
        self,
        time_horizon,
        bandit: MultiArmedBandit,
    ):
        # Add fields
        super().__init__(time_horizon, bandit)
        self.reward_memory = np.zeros(len(bandit.arms))
        self.count_memory = np.zeros(len(bandit.arms))
        self.time_step = 0

    def beta_sample(self):
        # beta_vect = np.vectorize(beta)
        beta_dist = np.array(
            [
                beta(a + 1, b + 1)
                for a, b in zip(
                    self.reward_memory, self.count_memory - self.reward_memory
                )
            ],
            dtype=object,
        )
        self.beta_dist = beta_dist

    def give_pull(self):
        self.beta_sample()
        # best_arm = np.argmax(self.beta_dist.fit(np.random.random))
        best_arm = np.argmax(np.array([dist.rvs() for dist in self.beta_dist]))
        reward = self.bandit.pull(best_arm)
        self.reinforce(reward, best_arm)

    def reinforce(self, reward, arm):
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)

    def plot_arm_graph(self):
        counts = self.count_memory
        indices = np.arange(len(counts))

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.bar(indices, counts, color="skyblue", edgecolor="black")

        # Formatting
        plt.title("Counts per Category", fontsize=16)
        plt.xlabel("Arm", fontsize=14)
        plt.ylabel("Pull Count", fontsize=14)
        plt.grid(axis="y", linestyle="-")  # Add grid lines for the y-axis
        plt.xticks(
            indices, [f"Category {i + 1}" for i in indices], rotation=45, ha="right"
        )
        # plt.yticks(np.arange(0, max(counts) + 2, step=2))

        # Annotate the bars with the count values
        for i, count in enumerate(counts):
            plt.text(
                i,
                count + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

        # Tight layout to ensure there's no clipping of labels
        plt.tight_layout()

        # Show plot
        plt.show()


# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23, 0.55, 0.76, 0.44]))
    agent = ThompsonSamplingAgent(
        TIME_HORIZON, bandit
    )  ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
