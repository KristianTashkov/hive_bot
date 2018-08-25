from collections import Counter

from engine.hive_game import HiveGame
from zoo.players import ModelPlayer


def evaluate(checkpoint, opponent, num_games=50, max_moves=100, to_win=6):
    game = HiveGame(to_win)
    results = []
    with ModelPlayer(game, 0, is_training=False, checkpoint=checkpoint) as player:
        with (ModelPlayer(game, 1, is_training=False, checkpoint=opponent) if isinstance(opponent, str)
              else opponent(game, 1)) as opponent:
            for num_game in range(num_games):
                ratio = ((num_game / num_games) * 100)
                if ratio % 10 == 0:
                    print("Done {}%".format(ratio))

                game.reset()
                moves_count = 0
                while game.get_winner() is None and moves_count < max_moves:
                    if game.to_play == player.player_id:
                        player.play_move()
                    else:
                        opponent.play_move()
                    winner = game.get_winner()
                    if winner is not None:
                        results.append(winner)
                        break
                    moves_count += 1
    print("Winrate: ", len([x for x in results if x == player.player_id]) / num_games, ", details:", Counter(results))
