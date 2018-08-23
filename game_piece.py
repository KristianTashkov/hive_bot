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