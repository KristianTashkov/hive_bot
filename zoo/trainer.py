import os
import traceback
import numpy as np

from engine.hive_game import HiveGame
from zoo.model import ConvModel
from collections import Counter
import time

from zoo.players import ModelPlayer, RandomPlayer


def get_reward(game, for_player):
    winner = game.get_winner()
    turns_penalty = -game.turns_passed * 0.001
    if winner is None or winner == -1:
        return turns_penalty
    return (10.0 if winner == for_player else -10.0) + turns_penalty


REWARD_DECAY = 0.98
BATCH_SIZE = 200
PLAYER_ID = 0
OPPONENT_ID = 1


def get_discount_rewards(r):
    discounted_r = np.zeros_like(r)
    discounted_r[-1] = r[-1]
    new_reward = r[-1]
    for t in reversed(range(0, len(r) - 1)):
        new_reward *= REWARD_DECAY
        discounted_r[t] = new_reward + r[t]
    return discounted_r


def reset_game(game, random_players, to_win):
    game.reset()
    simulate_turns = np.random.randint(0, to_win * 5)
    while simulate_turns > 0 or game.to_play != PLAYER_ID:
        random_players[game.to_play].play_move(game)
        simulate_turns -= 1
    if game.get_winner() is not None:
        reset_game(game, random_players, to_win)


def simulate_games(model_cls=ConvModel, checkpoint=None, opponent_checkpoint=None, previous_checkpoint=None,
                   exit_win_rate=0.65, keep_draw_ratio=0.2, save_every=500,
                   log_every=50, max_rounds=100, to_win=6, experiment_name=None, save_dir=None):
    print("{} vs {}, to_win={}".format(checkpoint, opponent_checkpoint, to_win))
    game = HiveGame(to_win=to_win)
    results = []
    exception_in_last_runs = []
    experiment_name = experiment_name if experiment_name is not None else str(round(time.time()))
    random_players = [RandomPlayer(), RandomPlayer()]
    with model_cls(is_training=True, checkpoint=checkpoint, save_dir=save_dir) as model:
        with ModelPlayer(is_training=False, checkpoint=opponent_checkpoint, model_cls=model_cls) as opponent:
            with (ModelPlayer(is_training=False, checkpoint=previous_checkpoint, model_cls=model_cls)
                  if previous_checkpoint is not None
                  else RandomPlayer()) as previous_opponent:
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
                                    opponent.play_move(game)
                                else:
                                    previous_opponent.play_move(game)

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
                        for batch_index in range(len(game_history) // BATCH_SIZE + len(game_history) % BATCH_SIZE != 0):
                            model.propagate_reward(all_states[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE, ...],
                                                   all_allowed[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE, ...],
                                                   played_actions[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE],
                                                   discounted_rewards[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE])
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
                        traceback.print_exc()
                        exception_in_last_runs.append(True)
                        if np.sum(exception_in_last_runs[:10]) >= 5:
                            raise

                    exception_in_last_runs = exception_in_last_runs[-10:]
                    game_index += 1
                model.save(experiment_name, game_index)
    return game_index


def full_training(model_cls=ConvModel, checkpoint=None, previous_opponent=None,
                  save_every=500, log_every=50, to_win_start=4, directory=None):
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
