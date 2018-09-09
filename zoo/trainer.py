import os
import traceback
import numpy as np

from engine.hive_game import HiveGame
from zoo.evaluate import evaluate
from zoo.model import ConvModel
from collections import Counter
import time

from zoo.players import ModelPlayer, RandomPlayer, AggressivePlayer


def get_reward(game, for_player):
    winner = game.get_winner()
    if winner is None or winner == -1:
        return -0.5
    return 1.0 if winner == for_player else -1.0


REWARD_DECAY = 0.98
BATCH_SIZE = 200
MAX_OBSERVATIONS = 2500


class Observation:
    def __init__(self, state, action):
        self.state = state
        self.action = action
        self.predicted_reward = state['reward']
        self.real_reward = 0


def train_step(player, observations):
    observations = np.random.choice(observations, min(BATCH_SIZE, len(observations)))
    all_states = np.vstack([x.state['board'] for x in observations])
    all_allowed = np.vstack([x.state['allowed_actions'] for x in observations])
    played_actions = np.array([x.action for x in observations])
    predicted_rewards = np.array([x.predicted_reward for x in observations])
    real_rewards = np.array([x.real_reward for x in observations])
    player.model.train_model(all_states, all_allowed, played_actions, real_rewards, real_rewards - predicted_rewards)


def simulate_games(model_cls=ConvModel, checkpoint=None, save_every=500,
                   log_every=50, max_turns=100, to_win=6):
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
                game = HiveGame(to_win=to_win)
                if np.random.rand() < 0.10:
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
                    state, action_id, action = player.play_move(game)
                    turns_remaining -= 1

                    if action is not None:
                        move_histories[player_id].append(Observation(state, action_id))

                for player_id, move_history in enumerate(move_histories):
                    moves_count = len(move_history)
                    if moves_count == 0:
                        continue
                    reward = get_reward(game, player_id)
                    move_history[moves_count - 1].real_reward = reward
                    for move_index in reversed(range(moves_count - 1)):
                        move_history[move_index].real_reward = (
                            move_history[move_index + 1].real_reward * REWARD_DECAY)
                for player_id in range(2):
                    observations.extend(move_histories[player_id])

                observations = observations[-MAX_OBSERVATIONS:]
                if len(observations) >= BATCH_SIZE and game_index % 5 == 0:
                    train_step(player, observations)

                winner = game.get_winner()
                results.append(winner if winner is not None else -1)
                if game_index != 0 and game_index % log_every == 0:
                    print(game_index, Counter(np.array(results)[-log_every:]),
                          start_position_collisions / game_index, '|')
                if game_index != 0 and game_index % save_every == 0:
                    player.model.save(experiment_name, game_index)
                    evaluate(checkpoint=os.path.join(save_dir, experiment_name, 'model.ckpt-' + str(game_index)),
                             opponent=AggressivePlayer, model_cls=model_cls,
                             to_win=to_win, max_moves=max_turns, no_log=True)
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
