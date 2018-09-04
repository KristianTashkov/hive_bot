import numpy as np
from collections import defaultdict

from engine.game_piece import GamePiece, GamePieceType


class HiveGame:
    NEIGHBORS_DIRECTION = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]
    TOTAL_PIECES_COUNT = 2 * sum(list(GamePieceType.PIECE_COUNT.values()))
    MAX_STACK_SIZE = 5

    def __init__(self, to_play=None, to_win=6):
        self.reset(to_play)
        self.to_win = to_win

    def reset(self, to_play=None):
        self.to_play = np.random.choice([0, 1], 1)[0] if to_play is None else to_play
        self._pieces = defaultdict(list)
        self._pieces_by_id = [{}, {}]
        self._neighbors = defaultdict(set)
        self._top_pieces = {}
        self.last_turn_pass = False
        self.game_drawed = False
        self.game_history = []
        self.turns_passed = 0

    def get_winner(self):
        if self.game_drawed:
            return -1
        queen1 = self.get_piece(0, 0)
        queen2 = self.get_piece(1, 0)
        queen1_taken = len(self._neighbors[queen1.position]) if queen1 is not None else 0
        queen2_taken = len(self._neighbors[queen2.position]) if queen2 is not None else 0

        if queen1_taken >= self.to_win:
            if queen2_taken >= self.to_win:
                return -1
            else:
                return 1
        elif queen2_taken >= self.to_win:
            return 0
        return None

    def play_action(self, action):
        if len(self.game_history) == 0:
            self.game_history.append(self.copy())
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
        self.turns_passed += 1
        self.game_history.append(self.copy())

    def all_pieces(self):
        return list(self._pieces_by_id[0].values()) + list(self._pieces_by_id[1].values())

    def deploy_piece(self, position, piece_type, force_color=None):
        color = force_color if force_color is not None else self.to_play
        new_id = GamePieceType.PIECE_INDEX[piece_type][
            len([x for x in self.player_pieces(color)
                 if x.piece_type == piece_type])]
        new_piece = GamePiece(self, color, piece_type, position, new_id)
        self.set_top_piece(position, new_piece)
        self._pieces_by_id[color][new_id] = new_piece

    def get_piece(self, color, id):
        return self._pieces_by_id[color].get(id)

    def get_stack(self, position):
        return self._pieces[position]

    def get_top_piece(self, position):
        return self._top_pieces.get(position)

    def set_top_piece(self, position, piece):
        old_piece = self.get_top_piece(position)
        self._pieces[position].append(piece)
        self._top_pieces[position] = piece
        for dx, dy in HiveGame.NEIGHBORS_DIRECTION:
            new_position = (position[0] + dx, position[1] + dy)
            self._neighbors[new_position].add(piece)
            if old_piece is not None:
                self._neighbors[new_position].remove(old_piece)

    def drop_top_piece(self, position):
        old_piece = self.get_top_piece(position)
        self._pieces[position] = self._pieces[position][:-1]
        piece = self._pieces[position][-1] if len(self._pieces[position]) > 0 else None
        self._top_pieces[position] = piece
        for dx, dy in HiveGame.NEIGHBORS_DIRECTION:
            new_position = (position[0] + dx, position[1] + dy)
            if piece is not None:
                self._neighbors[new_position].add(piece)
            self._neighbors[new_position].remove(old_piece)

    def player_pieces(self, color):
        return list(self._pieces_by_id[color].values())

    def neighbor_pieces(self, position):
        return self._neighbors[position]

    def is_connecting_piece(self, piece):
        if len(self.get_stack(piece.position)) > 1:
            return False
        neighbor_pieces = self._neighbors[piece.position]
        if len(neighbor_pieces) <= 1:
            return False
        used = {piece.position}
        queue = [list(neighbor_pieces)[0]]
        while len(queue) > 0:
            current, *queue = queue
            if current.position in used:
                continue
            used.add(current.position)
            queue.extend([x for x in self._neighbors[current.position] if x.position not in used])
        return len(used) != len([x for x in self._pieces.values() if len(x) > 0])

    def set_random_state(self, seed=None):
        np.random.seed(seed)
        self.reset()
        units_to_deploy = np.random.normal(HiveGame.TOTAL_PIECES_COUNT // 2,  HiveGame.TOTAL_PIECES_COUNT // 4)
        units_to_deploy = int(np.clip(units_to_deploy, 0, HiveGame.TOTAL_PIECES_COUNT))
        not_deployed_units = []
        free_positions = {(0, 0)}
        occupied_positions = set()
        other_color = self.to_play

        for color in range(2):
            for piece_type, count in GamePieceType.PIECE_COUNT.items():
                not_deployed_units.extend([(color, piece_type)] * count)
        for _ in range(units_to_deploy):
            forced_queen0 = (len(self._pieces_by_id[0].values()) == 3
                             and self.get_piece(0, 0) is None)
            forced_queen1 = (len(self._pieces_by_id[1].values()) == 3
                             and self.get_piece(1, 0) is None)
            forced_queens = [forced_queen0, forced_queen1]
            to_deploy = [x for x in not_deployed_units
                         if not forced_queens[x[0]] or x[1] == GamePieceType.QUEEN_BEE]
            if self.get_piece(other_color, 0) is None:
                to_deploy = [x for x in to_deploy if x[0] == other_color]
            color, piece_type = to_deploy[np.random.randint(0, len(to_deploy))]
            not_deployed_units.remove((color, piece_type))
            if len(occupied_positions) == 0:
                position = (0, 0)
            elif len(occupied_positions) == 1:
                position = (1, 0)
            else:
                to_deploy_positions = free_positions.copy()
                if piece_type == GamePieceType.BEETLE and self.get_piece(color, 0) is not None:
                    to_deploy_positions.update(occupied_positions)
                position = list(to_deploy_positions)[np.random.randint(0, len(to_deploy_positions))]

            self.deploy_piece(position, piece_type, force_color=color)
            occupied_positions.add(position)
            if position in free_positions:
                free_positions.remove(position)
            for dx, dy in HiveGame.NEIGHBORS_DIRECTION:
                new_position = position[0] + dx, position[1] + dy
                if new_position not in occupied_positions:
                    free_positions.add(new_position)
            last_color = color
            other_color = (last_color + 1) % 2
        self.to_play = other_color
        if self.get_winner() is not None:
            self.set_random_state(seed + 1 if seed is not None else None)

    def unique_id(self):
        unique_id = str(self.to_play)
        unique_id += str(self.last_turn_pass)

        all_pieces = []
        for color in range(2):
            for id in range(HiveGame.TOTAL_PIECES_COUNT // 2):
                piece = self.get_piece(color, id)
                all_pieces.append(piece)
        deployed_pieces = [x for x in all_pieces if x is not None]
        if len(deployed_pieces) == 0:
            return unique_id
        first_piece_position = deployed_pieces[0].full_position()
        for piece in all_pieces:
            if piece is None:
                unique_id += "_missing_"
            else:
                full_position = piece.full_position()
                unique_id += str((full_position[0] - first_piece_position[0],
                                  full_position[1] - first_piece_position[1],
                                  full_position[2] - first_piece_position[2]))
        return unique_id

    def copy_to(self, other):
        other.reset()
        other.to_play = self.to_play
        other.last_turn_pass = self.last_turn_pass
        other.game_drawed = self.game_drawed
        other.turns_passed = self.turns_passed
        other.to_win = self.to_win
        other.game_history = self.game_history
        for position, pieces in self._pieces.items():
            for piece in pieces:
                other.deploy_piece(piece.position, piece.piece_type, piece.color)

    def copy(self):
        game = HiveGame()
        self.copy_to(game)
        return game

