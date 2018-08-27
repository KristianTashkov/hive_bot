import numpy as np
from collections import defaultdict

from engine.game_piece import GamePiece, GamePieceType


class HiveGame:
    NEIGHBORS_DIRECTION = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]
    BOARD_SIZE = 2 * np.sum(list(GamePieceType.PIECE_COUNT.values()))
    MAX_STACK_SIZE = 5

    def __init__(self, to_play=None, to_win=6):
        self.reset(to_play)
        self.to_win = to_win

    def reset(self, to_play=None):
        self.to_play = np.random.choice([0, 1], 1)[0] if to_play is None else to_play
        self._pieces = defaultdict(list)
        self._pieces_by_id = [{}, {}]
        self.last_turn_pass = False
        self.game_drawed = False
        self.game_history = []
        self.turns_passed = 0
        self.internal_cache = {}

    def get_winner(self):
        queen1 = self.get_piece(0, 0)
        queen2 = self.get_piece(1, 0)
        if None in [queen1, queen2]:
            return None
        if self.game_drawed:
            return -1
        queen1_taken = len(self.neighbor_pieces(queen1.position))
        queen2_taken = len(self.neighbor_pieces(queen2.position))

        if queen1_taken == self.to_win:
            if queen2_taken == self.to_win:
                return -1
            else:
                return 1
        elif queen2_taken == self.to_win:
            return 0
        return None

    def play_action(self, action):
        if action is not None:
            action.debug = True
            if not action.can_be_played(self):
                raise ValueError("Action can't be played" + str(action))
            action.debug = False
            action.activate()
            self.last_turn_pass = False
        else:
            if self.last_turn_pass:
                self.game_drawed = True
            self.last_turn_pass = True
        self.to_play = (self.to_play + 1) % 2
        self.game_history.append(self.copy())
        self.turns_passed += 1
        self.internal_cache = {}

    def all_pieces(self):
        return list(self._pieces_by_id[0].values()) + list(self._pieces_by_id[1].values())

    def deploy_piece(self, position, piece_type):
        new_id = GamePieceType.PIECE_INDEX[piece_type][
            len([x for x in self.player_pieces(self.to_play)
                 if x.piece_type == piece_type])]
        new_piece = GamePiece(self, self.to_play, piece_type, position, new_id)
        self.set_top_piece(position, new_piece)
        self._pieces_by_id[self.to_play][new_id] = new_piece

    def get_piece(self, color, id):
        return self._pieces_by_id[color].get(id)

    def get_stack(self, position):
        return self._pieces[position]

    def get_top_piece(self, position):
        if len(self._pieces[position]) == 0:
            return None
        return self._pieces[position][-1]

    def set_top_piece(self, position, piece):
        self._pieces[position].append(piece)

    def drop_top_piece(self, position):
        self._pieces[position] = self._pieces[position][:-1]

    def player_pieces(self, color):
        return list(self._pieces_by_id[color].values())

    def neighbor_pieces(self, position):
        cache_key = ('neighbor_pieces', position)
        result = self.internal_cache.get(cache_key)
        if result is not None:
            return result
        pieces = []
        for dx, dy in HiveGame.NEIGHBORS_DIRECTION:
            piece = self.get_top_piece((position[0] + dx, position[1] + dy))
            if piece is not None:
                pieces.append(piece)
        self.internal_cache[cache_key] = pieces
        return pieces

    def is_connecting_piece(self, piece):
        if len(self.get_stack(piece.position)) > 1:
            return False
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

    def copy(self):
        game_copy = HiveGame()
        game_copy.to_play = self.to_play
        for position, pieces in self._pieces.items():
            for piece in pieces:
                new_piece = piece.copy(game_copy)
                game_copy._pieces[position].append(new_piece)
                game_copy._pieces_by_id[new_piece.color][new_piece.id] = new_piece

        game_copy.last_turn_pass = self.last_turn_pass
        game_copy.game_drawed = self.game_drawed
        return game_copy


