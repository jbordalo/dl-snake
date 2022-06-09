import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

from snake_game import SnakeGame
from game_demo import plot_board

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

train_episodes = 300


def train(replay_memory, model, target_model):
    discount_factor = 0.618
    batch_size = 64 * 2

    mini_batch = random.sample(replay_memory, batch_size)

    # get list of states
    current_states = np.array([transition[0] for transition in mini_batch])
    # predict expected discounted return of each action for each state in the mini-batch
    current_qs_list = model.predict(current_states)

    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)
    X = []
    Y = []

    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward
        # copy the current estimated expected return for all the actions
        current_qs = current_qs_list[index]
        # only change value for the action taken
        current_qs[action] = max_future_q

        X.append(observation)
        Y.append(current_qs)

    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def agent(state_shape, action_shape):
    learning_rate = 0.001

    inputs = keras.layers.Input(shape=state_shape, name='inputs')

    layer = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)

    layer = keras.layers.GlobalAveragePooling2D()(layer)
    layer = keras.layers.Dense(action_shape, activation='linear')(layer)

    model = keras.models.Model(inputs=inputs, outputs=layer)

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    model.summary()

    return model


if __name__ == '__main__':
    WIDTH = HEIGHT = 14
    BORDER = 9
    STATE_SHAPE = (32, 32, 3)
    game = SnakeGame(width=WIDTH, height=HEIGHT, border=BORDER, grass_growth=0.05)

    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.01
    MIN_REPLAY_SIZE = 1000
    MEMORY_SIZE = 50000

    model = agent(STATE_SHAPE, 3)
    target_model = agent(STATE_SHAPE, 3)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=MEMORY_SIZE)
    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation, _, _, _ = game.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            game.board_state()

            random_number = np.random.rand()
            if random_number <= epsilon:  # EXPLORATION
                # action can be -1, 0 or 1
                action = np.random.randint(0, 3) - 1
            else:  # EXPLOITATION
                predicted = model.predict(observation.reshape(-1, 32, 32, 3))  # .flatten()
                action = np.argmax(predicted)
            new_observation, reward, done, info = game.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])

            # train if we have enough examples
            if len(replay_memory) >= MIN_REPLAY_SIZE and (steps_to_update_target_model % 4 == 0 or done):
                train(replay_memory, model, target_model)
            observation = new_observation
            # if episode == 1:
            #     plot_board(f"{episode}{steps_to_update_target_model}.png", observation, action)
            total_training_rewards += reward
            if done:
                print(f"Rewards: {total_training_rewards} after n steps = {episode} with final reward = {reward}")
                total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

