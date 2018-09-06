import numpy as np

from engine.actions import create_all_actions
from zoo.model import ConvModel


class Player:
    def __init__(self):
        self.all_actions = create_all_actions()

    def _choose_from_actions(self, game, actions, state):
        raise NotImplemented()

    def _play_move(self, game, available_actions, state=None):
        if len(available_actions) == 0:
            game.play_action(None)
            return state, -1, None

        player_id = game.to_play
        _, taken_positions, losing_positions = game.queens_important_positions()

        non_losing_available_actions = [(index, x) for index, x in available_actions
                                        if (x.uniform_representation()[2] not in losing_positions[player_id] or
                                            x.uniform_representation()[1] in taken_positions[player_id])]
        if len(non_losing_available_actions) == 0:
            action_id, action = available_actions[0]
        else:
            winning_actions = [(index, x) for index, x in available_actions
                               if (x.uniform_representation()[2] in losing_positions[(player_id + 1) % 2] and
                                   x.uniform_representation()[1] not in taken_positions[(player_id + 1) % 2])]
            if len(winning_actions) > 0:
                action_id, action = winning_actions[0]
            else:
                action_id, action = self._choose_from_actions(game, non_losing_available_actions, state)
        game.play_action(action)
        return state, action_id, action

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ModelPlayer(Player):
    def __init__(self, *args, model_cls=ConvModel, **kwargs):
        super().__init__()
        self.model = model_cls(self.all_actions, *args, **kwargs)

    def __enter__(self):
        self.model.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.__exit__(exc_type, exc_val, exc_tb)

    def _choose_from_actions(self, game, actions, state):
        allowed_indexes = {x[0] for x in actions}
        for index in range(len(self.all_actions)):
            state['allowed_actions'][0][index] = 1 if index in allowed_indexes else 0
        return self.model.choose_action(state)

    def play_move(self, game):
        state = self.model.get_state(game)
        available_actions = [(index, x) for index, x in enumerate(self.all_actions)
                             if state['allowed_actions'][0][index] == 1]
        return self._play_move(game, available_actions, state)


class RandomPlayer(Player):
    def play_move(self, game):
        common_data = {}
        available_actions = [x for x in enumerate(self.all_actions) if x[1].can_be_played(game, common_data)]
        self._play_move(game, available_actions)

    def _choose_from_actions(self, game, actions, state):
        return actions[np.random.choice(range(len(actions)), 1)[0]]


class AggressivePlayer(RandomPlayer):
    def _choose_from_actions(self, game, actions, state):
        player_id = game.to_play
        free_positions, taken_positions, _ = game.queens_important_positions()
        aggresive_actions = [(index, x) for index, x in actions
                             if (x.uniform_representation()[2] in free_positions[(player_id + 1) % 2] and
                                 x.uniform_representation()[1] not in taken_positions[(player_id + 1) % 2])]
        if len(aggresive_actions) > 0:
            return aggresive_actions[0]
        non_relieving_actions = [(index, x) for index, x in actions
                                 if x.uniform_representation()[1] not in taken_positions[(player_id + 1) % 2]]
        if len(non_relieving_actions) > 0:
            actions = non_relieving_actions
        return actions[np.random.choice(range(len(actions)), 1)[0]]