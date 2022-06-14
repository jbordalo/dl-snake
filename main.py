import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cityblock

from animator import Animator
from snake_game import SnakeGame

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

train_episodes = 10000


def train(replay_memory, model, target_model, batch_size=64 * 2):
    discount_factor = 0.95
    # batch_size = 64 * 2
    #     learning_rate = float(model.optimizer._decayed_lr(tf.float32)) * 10
    learning_rate = 0.7
    # food_examples = [memory for memory in replay_memory if memory[2] >= 1.0]
    # death_examples = [memory for memory in replay_memory if memory[4]]
    #
    # mini_batch = random.sample(food_examples, min(int(batch_size / 4), len(food_examples)))
    # mini_batch += random.sample(death_examples, min(int(batch_size / 4), len(death_examples)))
    mini_batch = random.sample([memory for memory in replay_memory], batch_size)

    # get list of states
    current_states = np.array([transition[0] for transition in mini_batch])
    # predict expected discounted return of each action for each state in the mini-batch
    current_qs_list = model.predict(current_states)

    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)
    X = []
    Y = []

    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        action_index = action + 1

        # copy the current estimated expected return for all the actions
        current_qs = current_qs_list[index]
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])

            # if reward >= 1.0:
            # print(f"reward: {reward} pred future: {np.max(future_qs_list[index])} final: {max_future_q}")
            # print(f"prev q: {current_qs} future q: {(1 - learning_rate) * current_qs[action_index] + learning_rate * max_future_q} action: {action}")
            # only change value for the action taken
            current_qs[action_index] = (1 - learning_rate) * current_qs[action_index] + learning_rate * max_future_q
        else:
            # print(f"done q pred: {current_qs_list[index]} action: {action} reward: {reward}")
            # print(f"new q: {(1 - learning_rate) * current_qs_list[index][action_index] + learning_rate * reward}")
            max_future_q = reward
            # only change value for the action taken
            current_qs[action_index] = max_future_q

        X.append(observation)
        Y.append(current_qs)

    model.fit(np.array(X), np.array(Y), batch_size=64, verbose=1, shuffle=True)


def agent(state_shape, action_shape):
    learning_rate = 0.01

    inputs = keras.layers.Input(shape=state_shape, name='inputs')

    # layer = keras.layers.Conv2D(16, (8, 8), strides=(2, 2), padding="same", activation="relu")(inputs)
    # layer = keras.layers.BatchNormalization()(layer)
    #     layer = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform")(inputs)
    #     layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform")(inputs)
    layer = keras.layers.BatchNormalization()(layer)
    # layer = keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform")(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform")(layer)
    layer = keras.layers.BatchNormalization()(layer)
    # layer = keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)
    #
    # layer = keras.layers.GlobalAveragePooling2D()(layer)
    layer = keras.layers.Flatten()(layer)
    # layer = keras.layers.Dense(256, activation='relu', kernel_initializer="he_uniform")(layer)
    layer = keras.layers.Dense(128, activation='relu', kernel_initializer="he_uniform")(layer)
    layer = keras.layers.Dense(64, activation='relu', kernel_initializer="he_uniform")(layer)
    layer = keras.layers.Dense(32, activation='relu', kernel_initializer="he_uniform")(layer)
    layer = keras.layers.Dense(16, activation='relu', kernel_initializer="he_uniform")(layer)
    # layer = keras.layers.Dense(8, activation='relu', kernel_initializer="he_uniform")(layer)
    layer = keras.layers.Dense(action_shape, activation='linear')(layer)

    model = keras.models.Model(inputs=inputs, outputs=layer)

    model.compile(loss="huber",
                  optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.00005),
                  metrics=['mse'])

    model.summary()

    return model


def fake_grow_snake(head, d):
    """add one position to snake head (0=up, 1=right, 2=down, 3=left)"""
    y, x = head
    if d == 0:
        y = y - 1
    elif d == 1:
        x = x + 1
    elif d == 2:
        y = y + 1
    else:
        x = x - 1
    return [y, x]


def overflow_direction(d):
    if d < 0:
        return 3
    elif d > 3:
        return 0
    else:
        return d


def argmin(arr):
    arr = np.array(arr)
    candidates = np.arange(arr.size)
    candidates = candidates[arr == np.min(arr)]
    return np.random.choice(candidates)


def get_greedy_action():
    score, apple, head, tail, direction = game.get_state()
    dist = [cityblock(np.array(apple[0]), fake_grow_snake(head, overflow_direction(direction + d))) for d in range(-1, 2)]
    return argmin(dist) - 1


def get_straight_only_action():
    return 0


def generate_examples(n_examples, log=True):
    for episode in range(n_examples):
        # animator = Animator()
        total_training_rewards = 0
        observation, _, _, _ = game.reset()
        done = False
        while not done:
            action = get_greedy_action()
            new_observation, reward, done, info = game.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])
            observation = new_observation
            # animator.add_to_animation(observation)
            total_training_rewards += reward
            if done:
                if log:
                    print(f"Rewards: {total_training_rewards:.1f} after n steps = {episode} with final score = {info['score']:.1f}")
                total_training_rewards += 1
                # animator.save_animation(f"Example_{episode}")
                break


def generate_near_reward_or_ending_examples(n_examples, policy):
    for episode in range(n_examples):
        observation, _, _, _ = game.reset()
        memories = []
        done = False
        while not done:
            action = policy()
            new_observation, reward, done, info = game.step(action)
            memories.append([observation, action, reward, new_observation, done])
            if done or reward >= 1:
                replay_memory.extend(memories[-3:])
            observation = new_observation


if __name__ == '__main__':
    WIDTH = HEIGHT = 14
    BORDER = 9
    STATE_SHAPE = (32, 32, 3)
    game = SnakeGame(width=WIDTH, height=HEIGHT, border=BORDER, grass_growth=0.0005, max_grass=0.05, food_amount=1)

    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.0001
    decay = 0.001
    MIN_REPLAY_SIZE = 1000
    MEMORY_SIZE = 50000

    model = agent(STATE_SHAPE, 3)
    target_model = agent(STATE_SHAPE, 3)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=MEMORY_SIZE)
    generate_examples(5000)
    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        if episode > 9950 or episode % 100 == 0:
            animator = Animator()
        total_training_rewards = 0
        # generate_examples(1, False)
        # generate_near_reward_or_ending_examples(5, get_straight_only_action)
        # generate_near_reward_or_ending_examples(20, get_greedy_action)
        observation, _, _, _ = game.reset()
        done = False
        n_steps = 0
        while not done and n_steps < 1000:
            n_steps += 1
            steps_to_update_target_model += 1
            # game.board_state()

            random_number = np.random.rand()
            if random_number <= epsilon:  # EXPLORATION
                # action can be -1, 0 or 1
                # action = get_action()

                action = np.random.randint(0, 3) - 1
            else:  # EXPLOITATION
                predicted = model.predict(observation.reshape(-1, 32, 32, 3))
                action = np.argmax(predicted) - 1
            new_observation, reward, done, info = game.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])

            # train if we have enough examples
            if len(replay_memory) >= MIN_REPLAY_SIZE and (steps_to_update_target_model % 4 == 0 or done):
                train(replay_memory, model, target_model)
            observation = new_observation

            # plot_board(f"{episode}{steps_to_update_target_model}.png", observation, action)
            if episode > 9950 or episode % 100 == 0:
                animator.add_to_animation(observation)
            total_training_rewards += reward
            if done:
                print(f"Episode: {episode} Rewards: {total_training_rewards:.1f} after n steps = {n_steps} with final score = {info['score']:.1f}")
                print(f"q pred: {model.predict(observation.reshape(-1, 32, 32, 3))} action: {action} random: {random_number <= epsilon}")
                total_training_rewards += 1
                if episode > 9950 or episode % 100 == 0:
                    animator.save_animation(f"Episode_{episode}")

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
