import numpy as np
from collections import Counter

from engine.hive_game import HiveGame
from zoo.model import ConvModel
from zoo.players import ModelPlayer


def evaluate(checkpoint, opponent, num_games=50, max_moves=200, to_win=6, model_cls=ConvModel):
    print("Evaluating ", checkpoint)
    all_games = []
    with (ModelPlayer(is_training=False, checkpoint=checkpoint, random_move_prob=0, aggresive_move_prob=0,
                      model_cls=model_cls) if isinstance(checkpoint, str)
            else opponent()) as player:
        with (ModelPlayer(is_training=False, checkpoint=opponent, random_move_prob=0, aggresive_move_prob=0,
                          model_cls=model_cls) if isinstance(opponent, str)
              else opponent()) as opponent:
            for num_game in range(num_games):
                np.random.seed(num_game * 1000)
                try:
                    game = HiveGame(to_win=to_win)
                    moves_count = 0
                    while game.get_winner() is None and moves_count < max_moves:
                        if game.to_play == 0:
                            state, action_id, action = player.play_move(game)
                            #state_reward = player.evaluate_state(state, action_id)
                        else:
                            opponent.play_move(game)
                        moves_count += 1
                    all_games.append(game)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    import traceback
                    traceback.print_exc()
    print("Winrate: ", len([x for x in all_games if x.get_winner() == 0]) / num_games, ", details:",
          Counter([x.get_winner() for x in all_games]))
    return all_games
