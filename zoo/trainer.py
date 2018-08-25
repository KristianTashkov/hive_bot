import numpy as np
from engine.actions import create_all_actions
from engine.hive_game import HiveGame
from zoo.model import Model
from collections import Counter


def get_reward(game, for_player):
    winner = game.get_winner()
    if winner is None or winner == -1:
        return 0
    return 10 if winner == for_player else -10


def simulate_games():
    game = HiveGame()
    all_actions = create_all_actions(game, 1)
    results = []
    with Model(game, 0) as model:
        for i in range(1, 2):
            turns_remaining = 100
            game_history = []
            while game.get_winner() is None and turns_remaining < 0:
                state = model.get_state(game)
                to_play = game.to_play
                action_id = -1

                if to_play == 1:
                    action = np.random.choice([x for x in all_actions if x.can_be_played()], 1)[0]
                else:
                    action_id, action = model.choose_action(state, is_training=True)

                game.play_action(action)

                if to_play == 0:
                    reward = get_reward(game, to_play)
                    model.propagate_reward(state, action_id, reward)
                    game_history.append()
                turns_remaining -= 1
            winner = game.get_winner()
            print("Winner:", winner)
            results.append(winner if winner is not None else -1)
            if i % 10 == 0:
                print(Counter(np.array(results)[-10:]))

    return game

