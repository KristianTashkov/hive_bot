import os
import sys

import numpy as np

from engine.game_piece import GamePieceType
from engine.hive_game import HiveGame
from zoo.model import Model
from collections import Counter
import time

from zoo.players import ModelPlayer, RandomPlayer


def get_reward(game, for_player):
    winner = game.get_winner()
    turns_penalty = -game.turns_passed * 0.001
    if winner is None:
        return turns_penalty
        #player_queen = game.get_piece(for_player, 0)
        #opponent_queen = game.get_piece((for_player + 1) % 2, 0)
        #if None in [player_queen, opponent_queen]:
        #    return 0
        #player_taken = len(game.neighbor_pieces(player_queen.position))
        #opponent_taken = len(game.neighbor_pieces(opponent_queen.position))
        #return opponent_taken - player_taken + turns_penalty
    if winner == -1:
        return turns_penalty
    return (10.0 if winner == for_player else -10.0) + turns_penalty


gamma = 0.98
batch_size = 200


def get_discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    discounted_r[-1] = r[-1]
    new_reward = r[-1]
    for t in reversed(range(0, len(r) - 1)):
        new_reward *= gamma
        discounted_r[t] = new_reward + r[t]
    return discounted_r


PLAYER_ID = 0
OPPONENT_ID = 1


def reset_game(game, random_players, to_win):
    game.reset()
    simulate_turns = np.random.randint(0, to_win * 5)
    while simulate_turns > 0 or game.to_play != PLAYER_ID:
        random_players[game.to_play].play_move()
        simulate_turns -= 1
    if game.get_winner() is not None:
        reset_game(game, random_players, to_win)


def simulate_games(model_cls=Model, checkpoint=None, opponent_checkpoint=None, previous_checkpoint=None,
                   exit_win_rate=0.65, keep_draw_ratio=0.2, save_every=500,
                   log_every=50, max_rounds=100, to_win=6, experiment_name=None, save_dir=None):
    print("{} vs {}, to_win={}".format(checkpoint, opponent_checkpoint, to_win))
    game = HiveGame(to_win=to_win)
    results = []
    exception_in_last_runs = []
    experiment_name = experiment_name if experiment_name is not None else str(round(time.time()))
    random_players = [RandomPlayer(game, 0), RandomPlayer(game, 1)]
    with model_cls(game, PLAYER_ID, is_training=True, checkpoint=checkpoint, save_dir=save_dir) as model:
        with ModelPlayer(game, OPPONENT_ID, is_training=False, checkpoint=opponent_checkpoint, model_cls=model_cls) as opponent:
            with (ModelPlayer(game, OPPONENT_ID, is_training=False, checkpoint=previous_checkpoint, model_cls=model_cls)
                  if previous_checkpoint is not None
                  else RandomPlayer(game, OPPONENT_ID)) as previous_opponent:
                game_index = 0
                while (len(results) < 500 or
                       results[-100:].count(PLAYER_ID) / 100 < exit_win_rate or
                       results[-100:].count(-1) / 100 > keep_draw_ratio):
                    try:
                        reset_game(game, random_players, to_win)
                        rounds_remaining = max_rounds
                        game_history = []
                        discounted_rewards = []
                        old_reward = 0
                        while game.get_winner() is None and rounds_remaining > 0:
                            state = model.get_state(game)
                            action_id, action = model.choose_action(state)
                            game.play_action(action)
                            if game.get_winner() is None:
                                if np.random.random() < 0.8:
                                    opponent.play_move()
                                else:
                                    previous_opponent.play_move()

                            reward = get_reward(game, PLAYER_ID)
                            game_history.append([state, action_id, reward - old_reward])
                            old_reward = reward
                            discounted_rewards = get_discount_rewards([x[2] for x in game_history])
                            for j in range(len(game_history)):
                                game_history[j][2] = discounted_rewards[j]

                            rounds_remaining -= 1
                        all_states = np.vstack([x[0]['board'] for x in game_history])
                        all_allowed = np.vstack([x[0]['allowed_actions'] for x in game_history])
                        played_actions = np.array([x[1] for x in game_history])
                        for batch_index in range(len(game_history) // batch_size + len(game_history) % batch_size != 0):
                            model.propagate_reward(all_states[batch_index * batch_size:(batch_index + 1) * batch_size, ...],
                                                   all_allowed[batch_index * batch_size:(batch_index + 1) * batch_size, ...],
                                                   played_actions[batch_index * batch_size:(batch_index + 1) * batch_size],
                                                   discounted_rewards[batch_index * batch_size:(batch_index + 1) * batch_size])
                        winner = game.get_winner()
                        results.append(winner if winner is not None else -1)
                        if game_index != 0 and game_index % log_every == 0:
                            print(game_index, Counter(np.array(results)[-log_every:]))
                        if game_index != 0 and game_index % save_every == 0:
                            model.save(experiment_name, game_index)
                        exception_in_last_runs.append(False)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        exception_in_last_runs.append(True)
                        if np.sum(exception_in_last_runs[:10]) >= 5:
                            raise

                    exception_in_last_runs = exception_in_last_runs[-10:]
                    game_index += 1
                model.save(experiment_name, game_index)
    return game_index


def full_training(model_cls=Model, checkpoint=None, previous_opponent=None, save_every=500, log_every=50, to_win_start=4, directory=None):
    directory = directory if directory is not None else 'D:\\code\\hive\\checkpoints\\'
    experiment_index = 0
    main_name = str(round(time.time()))

    to_win = to_win_start
    while True:
        experiment_name = "{}_{}".format(main_name, experiment_index)
        last_game_index = simulate_games(model_cls=model_cls, checkpoint=checkpoint, opponent_checkpoint=checkpoint,
                                         previous_checkpoint=previous_opponent,
                                         save_every=save_every, log_every=log_every, max_rounds=50,
                                         to_win=to_win, experiment_name=experiment_name, save_dir=directory)
        previous_opponent = checkpoint
        checkpoint = os.path.join(directory, experiment_name, 'model.ckpt-' + str(last_game_index))
        experiment_index += 1
        if to_win < 6:
            to_win += 1


if __name__ == 'main':
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    to_win_start = sys.argv[2] if len(sys.argv) > 2 else 3
    full_training(checkpoint=checkpoint, to_win_start=3)
