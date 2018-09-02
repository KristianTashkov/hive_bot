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
    if winner == -1:
        return turns_penalty
    return (10.0 if winner == for_player else -10.0) + turns_penalty


REWARD_DECAY = 0.95
BATCH_SIZE = 200
MAX_OBSERVATIONS = 10000


class Observation:
    def __init__(self, state, action):
        self.state = state
        self.action = action
        self.reward = 0


def train_step(player, observations):
    observations = np.random.choice(observations, BATCH_SIZE)
    all_states = np.vstack([x.state['board'] for x in observations])
    all_allowed = np.vstack([x.state['allowed_actions'] for x in observations])
    played_actions = np.array([x.action for x in observations])
    rewards = np.array([x.reward for x in observations])
    player.model.propagate_reward(
        all_states, all_allowed, played_actions, rewards)


def simulate_games(model_cls=ConvModel, checkpoint=None, save_every=500,
                   log_every=50, max_turns=200, to_win=6):
    game = HiveGame(to_win=to_win)
    results = []
    exception_in_last_runs = []
    experiment_name = str(round(time.time()))
    save_dir = 'D:\\code\\hive\\checkpoints\\'
    all_start_positions = set()
    start_position_collisions = 0
    with ModelPlayer(is_training=True, checkpoint=checkpoint, model_cls=model_cls, save_dir=save_dir) as player:
        game_index = 0
        observations = []
        while True:
            try:
                if np.random.rand() < 0.01:
                    game.reset()
                else:
                    game.set_random_state()
                    unique_id = game.unique_id()
                    if unique_id in all_start_positions:
                        start_position_collisions += 1
                    else:
                        all_start_positions.add(unique_id)
                turns_remaining = max_turns
                move_histories = [[], []]
                while game.get_winner() is None and turns_remaining > 0:
                    player_id = game.to_play
                    state = player.model.get_state(game)
                    action_id, action = player.model.choose_action(state)
                    game.play_action(action)
                    turns_remaining -= 1

                    if action is not None:
                        move_histories[player_id].append(Observation(state, action_id))
                    if len(observations) >= MAX_OBSERVATIONS:
                        train_step(player, observations)

                for player_id, move_history in enumerate(move_histories):
                    moves_count = len(move_history)
                    if moves_count == 0:
                        continue
                    reward = get_reward(game, player_id)
                    move_history[moves_count - 1].reward = reward
                    for move_index in reversed(range(moves_count - 1)):
                        move_history[move_index].reward = move_history[move_index + 1].reward * REWARD_DECAY
                for player_id in range(2):
                    observations.extend(move_histories[player_id])
                observations = observations[-MAX_OBSERVATIONS:]

                winner = game.get_winner()
                results.append(winner if winner is not None else -1)
                if game_index != 0 and game_index % log_every == 0:
                    print(game_index, Counter(np.array(results)[-log_every:]),
                          start_position_collisions / game_index, '|')
                if game_index != 0 and game_index % save_every == 0:
                    player.model.save(experiment_name, game_index)
                    evaluate(checkpoint=os.path.join(save_dir, experiment_name, 'model.ckpt-' + str(game_index)),
                             opponent=RandomPlayer, model_cls=model_cls,
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
