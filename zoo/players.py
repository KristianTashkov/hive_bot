import numpy as np

from engine.actions import create_all_actions
from zoo.model import ConvModel


class ModelPlayer:
    def __init__(self, *args, model_cls=ConvModel, **kwargs):
        self.model = model_cls(*args, **kwargs)

    def __enter__(self):
        self.model.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.__exit__(exc_type, exc_val, exc_tb)

    def play_move(self, game):
        state = self.model.get_state(game)
        _, action = self.model.choose_action(state)
        game.play_action(action)


class RandomPlayer:
    def __init__(self):
        self.all_actions = create_all_actions()

    def play_move(self, game):
        common_data = {}
        available_actions = [x for x in self.all_actions if x.can_be_played(game, common_data)]
        if len(available_actions) == 0:
            game.play_action(None)
            return

        player_id = game.to_play
        taken_positions, losing_positions = game.queens_important_positions()

        non_losing_available_actions = [x for x in available_actions
                                        if (x.uniform_representation()[2] not in losing_positions[player_id] or
                                            x.uniform_representation()[1] in taken_positions[player_id])]
        if len(non_losing_available_actions) == 0:
            action = available_actions[0]
        else:
            winning_actions = [x for x in available_actions
                               if (x.uniform_representation()[2] in losing_positions[(player_id + 1) % 2] and
                                   x.uniform_representation()[1] not in taken_positions[(player_id + 1) % 2])]
            if len(winning_actions) > 0:
                action = winning_actions[0]
            else:
                action = np.random.choice(non_losing_available_actions, 1)[0]
        game.play_action(action)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
