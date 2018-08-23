from itertools import product

import numpy as np
from collections import defaultdict

from actions import DeployAction, MoveAction


class GamePieceType:
    QUEEN_BEE = 'queen'
    BEETLE = 'beetle'
    GRASSHOPPER = 'grasshopper'
    SPIDER = 'spider'
    SOLDIER_ANT = 'ant'

    PIECE_COUNT = {
        QUEEN_BEE: 1,
        BEETLE: 2,
        GRASSHOPPER: 3,
        SPIDER: 2,
        SOLDIER_ANT: 3
    }


class GamePiece:
    def __init__(self, hive_game, color, piece_type, position):
        self.game = hive_game
        self.color = color
        self.piece_type = piece_type
        self.position = position

    def __repr__(self):
        return '{}[{}]({}, {})'.format(
            self.piece_type, self.color, self.position[0], self.position[1])


class HiveGame:
    NEIGHBORS_DIRECTION = [[0, -1], [0, 1], [-1, 0], [1, 0], [1, -1], [-1, 1]]
    BOARD_SIZE = 2 * np.sum(list(GamePieceType.PIECE_COUNT.values()))
    MAX_STACK_SIZE = 5

    def __init__(self):
        self.to_play = np.random.choice([0, 1], 1)[0]
        self._pieces = defaultdict(list)
        self.queens = [None, None]
        self.last_turn_pass = False
        self.game_drawed = False

    def get_winner(self):
        if None in self.queens:
            return None
        if self.game_drawed:
            return -1
        for queen in self.queens:
            if len(self.neighbor_pieces(queen.position)) == 6:
                return (queen.color + 1) % 2
        return None

    def play_action(self, action):
        if action is not None:
            assert action.can_be_played()
            action.activate()
            self.last_turn_pass = False
        else:
            if self.last_turn_pass:
                self.game_drawed = True
            self.last_turn_pass = True
        self.to_play = (self.to_play + 1) % 2

    def get_top_piece(self, position):
        if len(self._pieces[position]) == 0:
            return None
        return self._pieces[position][-1]

    def set_top_piece(self, position, piece):
        self._pieces[position].append(piece)

    def drop_top_piece(self, position):
        self._pieces[position] = self._pieces[position][:-1]

    def all_pieces(self):
        return [x for stack in self._pieces.values() for x in stack]

    def player_pieces(self, color):
        return [x for x in self.all_pieces() if x.color == color]

    def neighbor_pieces(self, position):
        pieces = []
        for dx, dy in HiveGame.NEIGHBORS_DIRECTION:
            piece = self.get_top_piece((position[0] + dx, position[1] + dy))
            if piece is not None:
                pieces.append(piece)
        return pieces

    def is_connecting_piece(self, piece):
        neighbor_pieces = self.neighbor_pieces(piece.position)
        if len(neighbor_pieces) <= 1:
            return False
        used = {piece.position}
        queue = [neighbor_pieces[0]]
        while len(queue) > 0:
            current, *queue = queue
            if current.position in used:
                continue
            used.add(current.position)
            neighbor_pieces = self.neighbor_pieces(current.position)
            queue.extend([x for x in neighbor_pieces if x.position not in used])
        return len(used) != len([x for x in self._pieces.values() if len(x) > 0])

    def all_actions(self):
        available_actions = []
        for piece_type in GamePieceType.PIECE_COUNT.keys():
            for x,y in product(range(-5, 5), range(-5, 5)):
                    deploy = DeployAction(self, self.to_play, piece_type, (x, y))
                    if deploy.can_be_played():
                        available_actions.append(deploy)
        for start_x, start_y in product(range(-5, 5), range(-5, 5)):
            for end_x, end_y in product(range(-5, 5), range(-5, 5)):
                move = MoveAction(self, (start_x, start_y), (end_x, end_y))
                if move.can_be_played():
                    available_actions.append(move)
        if len(available_actions) == 0:
            return [None]
        return available_actions

    def print_game(self, mark_space=None):
        extra_space = ''
        print(''.join([((' ' if len(str(x)) == 1 else '') + ' {} '.format(x)) for x in range(-5, 10)]))
        for y in range(-5, 5):
            line = (' ' if len(str(y)) == 1 else '') + str(y) + extra_space
            for x in range(-5, 10):
                piece = self.get_top_piece((x, y))
                if mark_space == (x, y):
                    current = 'xx'
                elif piece is None:
                    current = '__'
                else:
                    current = piece.piece_type[0] + str(piece.color)
                line += ' ' + current + ' '
            print(line)
            extra_space += '   '


