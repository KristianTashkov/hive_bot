import numpy as np
from collections import Counter

from engine.hive_game import HiveGame
from zoo.model import ConvModel
from zoo.players import ModelPlayer


def evaluate(checkpoint, opponent, num_games=100, max_moves=100, to_win=6, model_cls=ConvModel):
    game = HiveGame(to_win)
    results = []
    np.random.seed(17)
    with (ModelPlayer(is_training=False, checkpoint=checkpoint,
                      model_cls=model_cls) if isinstance(checkpoint, str)
            else opponent()) as player:
        with (ModelPlayer(is_training=False, checkpoint=opponent,
                          model_cls=model_cls) if isinstance(opponent, str)
              else opponent()) as opponent:
            for num_game in range(num_games):
                try:
                    ratio = ((num_game / num_games) * 100)
                    if ratio != 0 and ratio % 10 == 0:
                        print("Done {}% [{}]".format(
                            ratio, len([x for x in results if x == 0]) / len(results)))

                    game.reset()
                    moves_count = 0
                    while game.get_winner() is None and moves_count < max_moves:
                        if game.to_play == 0:
                            player.play_move(game)
                        else:
                            opponent.play_move(game)
                        moves_count += 1
                    results.append(game.get_winner())
                except KeyboardInterrupt:
                    raise
                except Exception:
                    import traceback
                    traceback.print_exc()
    print("Winrate: ", len([x for x in results if x == 0 ]) / num_games, ", details:", Counter(results))
