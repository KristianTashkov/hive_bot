import os

import numpy as np
from engine.hive_game import HiveGame
from zoo.model import Model
from collections import Counter
import time

from zoo.players import ModelPlayer


def get_reward(game, for_player):
    winner = game.get_winner()
    if winner is None or winner == -1:
        return 0.0
    return 10.0 if winner == for_player else -10.0


gamma = 0.99
batch_size = 10


def get_discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


PLAYER_ID = 0
OPPONENT_ID = 1


def simulate_games(checkpoint=None, opponent_checkpoint=None, num_games=100000, save_every=500, log_every=50,
                   max_rounds=100, to_win=6, experiment_name=None, save_dir=None):
    print("{} vs {}, to_win={}, num_games={}".format(checkpoint, opponent_checkpoint, to_win, num_games))
    game = HiveGame(to_win=to_win)
    results = []
    exception_in_last_runs = []
    experiment_name = experiment_name if experiment_name is not None else round(time.time())
    with Model(game, PLAYER_ID, is_training=True, checkpoint=checkpoint, save_dir=save_dir) as model:
        with ModelPlayer(game, 1, is_training=False, checkpoint=opponent_checkpoint) as opponent:
            for game_index in range(0, num_games + 1):
                try:
                    game.reset()
                    if game.to_play != PLAYER_ID:
                        opponent.play_move()
                    rounds_remaining = max_rounds
                    game_history = []
                    discounted_rewards = []
                    while game.get_winner() is None and rounds_remaining > 0:
                        state = model.get_state(game)
                        action_id, action = model.choose_action(state)
                        game.play_action(action)
                        if game.get_winner() is None:
                            opponent.play_move()

                        reward = get_reward(game, PLAYER_ID)
                        game_history.append([state, action_id, reward])
                        discounted_rewards = get_discount_rewards([x[2] for x in game_history])
                        for j in range(len(game_history)):
                            game_history[j][2] = discounted_rewards[j]

                            rounds_remaining -= 1
                    winner = game.get_winner()

                    if winner is not None:
                        all_states = np.vstack([x[0]['board'] for x in game_history])
                        all_allowed = np.vstack([x[0]['allowed_actions'] for x in game_history])
                        played_actions = np.array([x[1] for x in game_history])
                        for batch_index in range(len(game_history) // batch_size + len(game_history) % batch_size != 0):
                            model.propagate_reward(all_states[batch_index * batch_size:(batch_index + 1) * batch_size, ...],
                                                   all_allowed[batch_index * batch_size:(batch_index + 1) * batch_size, ...],
                                                   played_actions[batch_index * batch_size:(batch_index + 1) * batch_size],
                                                   discounted_rewards[batch_index * batch_size:(batch_index + 1) * batch_size])
                    results.append(winner if winner is not None else -1)
                    if game_index % log_every == 0:
                        print(game_index, Counter(np.array(results)[-log_every:]))
                    if game_index % save_every == 0:
                        model.save(experiment_name, game_index)
                    exception_in_last_runs.append(False)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    import traceback
                    print(e)
                    traceback.print_exc()
                    exception_in_last_runs.append(True)
                    if np.sum(exception_in_last_runs[:10]) >= 5:
                        raise

                exception_in_last_runs = exception_in_last_runs[-10:]
            if game_index % save_every != 0:
                model.save(experiment_name, game_index)
    return experiment_name


def full_training(checkpoint=None, series_games=10000, num_games=100000, save_every=500, log_every=50,
                  max_rounds=100, directory=None):
    directory = directory if directory is not None else 'D:\\code\\hive\\checkpoints\\'
    experiment_index = 0

    for to_win in [4, 5] + [6] * (num_games // series_games - 2):
        experiment_name = "{}_{}".format(str(round(time.time())), experiment_index)
        simulate_games(checkpoint=checkpoint, opponent_checkpoint=checkpoint,
                       num_games=series_games, save_every=save_every, log_every=log_every, max_rounds=max_rounds,
                       to_win=to_win, experiment_name=experiment_name, save_dir=directory)
        checkpoint = os.path.join(directory, experiment_name, 'model.ckpt-' + str(series_games))
        experiment_index += 1

