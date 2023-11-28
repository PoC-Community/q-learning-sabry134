import random
import gym
import numpy as np
import matplotlib.pyplot as plt

def init_q_table(x: int, y: int) -> np.ndarray:
    """
    This function must return a 2D matrix containing only zeros for values.
    """
    return np.zeros((x, y))

qTable = init_q_table(5, 4)
print("Q-Table:\n" + str(qTable))
assert(np.mean(qTable) == 0)

LEARNING_RATE = 0.05
DISCOUNT_RATE = 0.99

def q_function(q_table: np.ndarray, state: int, action: int, reward: int, newState: int) -> float:
    """
    This function implements the q_function equation.
    It returns the updated q-table.
    """
    current_q_value = q_table[state, action]
    max_future_q = np.max(q_table[newState, :])
    new_q_value = (1 - LEARNING_RATE) * current_q_value + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q)
    q_table[state, action] = new_q_value
    return q_table

q_table = init_q_table(5, 4)
q_table[0, 1] = q_function(q_table, state=0, action=1, reward=-1, newState=3)
print("Q-Table after action:\n" + str(q_table))
assert(q_table[0, 1] == -LEARNING_RATE), f"The Q function is incorrect: the value of qTable[0, 1] should be -{LEARNING_RATE}"

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

total_actions = env.action_space.n
assert(total_actions == 4), f"There are a total of four possible actions in this environment. Your answer is {total_actions}"

def random_action(env):
    return random.randint(0, total_actions - 1)

def game_loop(env: gym.Env, q_table: np.ndarray, state: int, action: int) -> tuple:
    observation, reward, done, info = env.step(action)
    q_table = q_function(q_table, state, action, reward, observation)
    return q_table, observation, done, reward

EPOCH = 20000
for i in range(EPOCH):
    state, info = env.reset()
    while True:
        action = random_action(env)
        q_table, state, done, reward = game_loop(env, q_table, state, action)
        if done:
            break

wins = 0.0
for i in range(100):
    state, info = env.reset()
    while True:
        action = np.argmax(q_table[state, :])
        _, state, done, reward = game_loop(env, q_table, state, action)
        if done:
            if reward > 0:
                wins += 1
            break

print(f"{round(wins / (i+1) * 100, 2)}% winrate")
print(wins)

plt.imshow(env.render())
env.close()
