import os
import traceback
import numpy as np

from engine.hive_game import HiveGame
from zoo.evaluate import evaluate
from zoo.model import ConvModel
from collections import Counter
import time

from zoo.players import ModelPlayer, RandomPlayer


def get_reward(game, for_player):
    winner = game.get_winner()
    turns_penalty = -game.turns_passed * 0.01
    if winner is None or winner == -1:
        return turns_penalty
    return (10.0 if winner == for_player else -10.0) + turns_penalty


REWARD_DECAY = 0.98
BATCH_SIZE = 200


def get_discount_rewards(r):
    discounted_r = np.zeros_like(r)
    discounted_r[-1] = r[-1]
    new_reward = r[-1]
    for t in reversed(range(0, len(r) - 1)):
        new_reward *= REWARD_DECAY
        discounted_r[t] = new_reward + r[t]
    return discounted_r


def simulate_games(model_cls=ConvModel, checkpoint=None, save_every=500,
                   log_every=50, max_rounds=100, to_win=6):
    game = HiveGame(to_win=to_win)
    results = []
    exception_in_last_runs = []
    experiment_name = str(round(time.time()))
    save_dir = 'D:\\code\\hive\\checkpoints\\'
    all_start_positions = set()
    start_position_collisions = 0
    with ModelPlayer(is_training=False, checkpoint=checkpoint, model_cls=model_cls, save_dir=save_dir) as player:
        game_index = 0
        while True:
            try:
                if np.random.rand() < 0.01:
                    game.reset()
                else:
                    game.set_random_state()
                    unique_id = game.unique_id()
                    if unique_id in all_start_positions:
                        start_position_collisions += 1
                        print("Collisions are: ", start_position_collisions / game_index)
                    else:
                        all_start_positions.add(unique_id)
                training_model_id = game.to_play
                rounds_remaining = max_rounds
                game_history = []
                discounted_rewards = []
                old_reward = 0
                while game.get_winner() is None and rounds_remaining > 0:
                    state = player.model.get_state(game)
                    action_id, action = player.model.choose_action(state)
                    game.play_action(action)
                    if game.get_winner() is None:
                        player.play_move(game)

                    if action is not None:
                        reward = get_reward(game, training_model_id)
                        game_history.append([state, action_id, reward - old_reward])
                        old_reward = reward
                        discounted_rewards = get_discount_rewards([x[2] for x in game_history])
                        for j in range(len(game_history)):
                            game_history[j][2] = discounted_rewards[j]

                        rounds_remaining -= 1
                if len(game_history) == 0:
                    continue
                all_states = np.vstack([x[0]['board'] for x in game_history])
                all_allowed = np.vstack([x[0]['allowed_actions'] for x in game_history])
                played_actions = np.array([x[1] for x in game_history])
                for batch_index in range(len(game_history) // BATCH_SIZE + len(game_history) % BATCH_SIZE != 0):
                    player.model.propagate_reward(all_states[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE, ...],
                                           all_allowed[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE, ...],
                                           played_actions[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE],
                                           discounted_rewards[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE])
                winner = game.get_winner()
                results.append(winner if winner is not None else -1)
                if game_index != 0 and game_index % log_every == 0:
                    print(game_index, Counter(np.array(results)[-log_every:]))
                if game_index != 0 and game_index % save_every == 0:
                    player.model.save(experiment_name, game_index)
                    evaluate(checkpoint=os.path.join(save_dir, experiment_name, 'model.ckpt-' + str(game_index)),
                             opponent=RandomPlayer,
                             to_win=to_win, no_log=True)
                exception_in_last_runs.append(False)
            except KeyboardInterrupt:
                raise
            except Exception:
                traceback.print_exc()
                exception_in_last_runs.append(True)
                if np.sum(exception_in_last_runs[:10]) >= 5:
                    raise

            game_index += 1
            exception_in_last_runs = exception_in_last_runs[-10:]
