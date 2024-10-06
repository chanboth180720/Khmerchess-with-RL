class KhmerChessBoard:
    def __init__(self):
        self.board = torch.zeros(8, 8, dtype=torch.int)
        self.initialize_board() 
        self.player = 1  # Player 1 starts the game
        self.first_moves = {}  # Track first moves for king and queen

    def initialize_board(self):
        self.board[0, :] = torch.tensor([4, 2, 3, 5, 6, 3, 2, 4], dtype=torch.int)
        self.board[2, :] = 1  # Pawns for Player 2
        self.board[5, :] = -1  # Pawns for Player 1
        self.board[7, :] = -torch.tensor([4, 2, 3, 6, 5, 3, 2, 4], dtype=torch.int)
        self.first_moves = {
            (1, 0, 4): True,  # White King
            (1, 0, 3): True,  # White Queen
            (-1, 7, 4): True, # Black King
            (-1, 7, 3): True  # Black Queen
        }

    def move_piece(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square

        if not self.is_valid_square(from_row, from_col) or not self.is_valid_square(to_row, to_col):
            return False

        if self.board[from_row, from_col] == 0:
            return False

        if self.board[to_row, to_col] * self.board[from_row, from_col] > 0:
            return False

        if not self.is_legal_move(from_square, to_square):
            return False

        piece_type = abs(self.board[from_row, from_col])
        
        # Special move logic for king
        if piece_type == 6:  # King
            if self.first_moves.get((self.board[from_row, from_col], from_row, from_col), False):
                if self.is_king_jump(from_square, to_square):
                    self.first_moves[(self.board[from_row, from_col], from_row, from_col)] = False
                else:
                    return False
            if self.is_in_check(self.player):
                return False
        
        # Special move logic for queen
        if piece_type == 5:  # Queen
            if self.first_moves.get((self.board[from_row, from_col], from_row, from_col), False):
                if self.is_queen_jump(from_square, to_square):
                    self.first_moves[(self.board[from_row, from_col], from_row, from_col)] = False
                else:
                    return False

        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = 0
        self.switch_player()
        return True

    def is_king_jump(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        direction = 1 if self.board[from_row, from_col] > 0 else -1

        # Check if the king is jumping like a knight to the second row
        if (abs(from_row - to_row), abs(from_col - to_col)) in [(1, 2), (2, 1)] and from_row == 0:
            if self.is_valid_square(to_row, to_col):
                if self.board[to_row, to_col] == 0:
                    return True

        return False

    def is_queen_jump(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        direction = 1 if self.board[from_row, from_col] > 0 else -1

        # Check if the queen is jumping two squares forward
        if from_col == to_col and abs(to_row - from_row) == 2 and from_row == 0:
            if self.board[to_row, to_col] == 0:
                return True

        return False

    def is_in_check(self, player):
        # Check if the king of the player is in check
        return self.check_check(player)

    def capture_piece(self, row, col):
        if self.is_opponent_piece(row, col, self.player):
            self.board[row, col] = 0

    def is_legal_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square

        if not self.is_valid_square(from_row, from_col) or not self.is_valid_square(to_row, to_col):
            return False
        if self.board[from_row, from_col] == 0:
            return False
        if self.board[to_row, to_col] * self.board[from_row, from_col] > 0:
            return False

        piece_type = abs(self.board[from_row, from_col])
        if piece_type == 1:  # Pawn
            return self.is_legal_pawn_move(from_square, to_square)
        elif piece_type == 2:  # Knight
            return self.is_legal_knight_move(from_square, to_square)
        elif piece_type == 3:  # Bishop
            return self.is_legal_bishop_move(from_square, to_square)
        elif piece_type == 4:  # Rook
            return self.is_legal_rook_move(from_square, to_square)
        elif piece_type == 5:  # Queen
            return self.is_legal_queen_move(from_square, to_square)
        elif piece_type == 6:  # King
            return self.is_legal_king_move(from_square, to_square)

        return False

    def is_legal_pawn_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        direction = 1 if self.board[from_row, from_col] > 0 else -1

        if to_col == from_col:
            if self.board[to_row, to_col] == 0:
                if to_row - from_row == direction:
                    return True
                if (from_row == 1 and direction == 1) or (from_row == 6 and direction == -1):
                    if to_row - from_row == 2 * direction and self.board[from_row + direction, from_col] == 0:
                        return True

        if abs(to_col - from_col) == 1 and to_row - from_row == direction:
            if self.is_opponent_piece(to_row, to_col, self.player):
                return True

        return False

    def is_legal_knight_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        return (abs(from_row - to_row), abs(from_col - to_col)) in [(1, 2), (2, 1)]

    def is_legal_bishop_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        from_piece = self.board[from_row, from_col]
        direction = 1 if from_piece > 0 else -1  # White moves up (1), black moves down (-1)

        # Check for one-square diagonal movement
        if (abs(from_row - to_row) == 1 and abs(from_col - to_col) == 1):
            dest_piece = self.board[to_row, to_col]
            if from_piece > 0:  # White bishop
                if dest_piece <= 0:  # Move to an empty square or capture black piece
                    return True
            else:  # Black bishop
                if dest_piece >= 0:  # Move to an empty square or capture white piece
                    return True

        # Check for one-square forward movement
        if from_col == to_col and (to_row - from_row) == direction:
            dest_piece = self.board[to_row, to_col]
            if from_piece > 0:  # White bishop
                if dest_piece <= 0:  # Move to an empty square or capture black piece
                    return True
            else:  # Black bishop
                if dest_piece >= 0:  # Move to an empty square or capture white piece
                    return True

        return False

    def is_legal_rook_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        if from_row != to_row and from_col != to_col:
            return False
        if from_row == to_row:
            step = 1 if to_col > from_col else -1
            for col in range(from_col + step, to_col, step):
                if self.board[from_row, col] != 0:
                    return False
        else:
            step = 1 if to_row > from_row else -1
            for row in range(from_row + step, to_row, step):
                if self.board[row, from_col] != 0:
                    return False
        return True

    def is_legal_queen_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        return (abs(from_row - to_row), abs(from_col - to_col)) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def is_legal_king_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        return max(abs(from_row - to_row), abs(from_col - to_col)) == 1
    
    def get_all_legal_moves(self, player):
        legal_moves = []
        for row in range(8):
            for col in range(8):
                if self.is_own_piece(row, col, player):
                    for to_row in range(8):
                        for to_col in range(8):
                            if self.is_legal_move((row, col), (to_row, to_col)):
                                legal_moves.append(((row, col), (to_row, to_col)))
        return legal_moves

    def is_valid_square(self, row, col):
        return 0 <= row < 8 and 0 <= col < 8

    def is_opponent_piece(self, row, col, player):
        return self.board[row, col] * player < 0

    def is_own_piece(self, row, col, player):
        return self.board[row, col] * player > 0

    def check_checkmate(self, player):
        if not self.check_check(player):
            return False

        if not self.get_all_legal_moves(player):
            return True

        for row in range(8):
            for col in range(8):
                if self.is_own_piece(row, col, player):
                    for to_row in range(8):
                        for to_col in range(8):
                            if self.is_legal_move((row, col), (to_row, to_col)):
                                piece = self.board[row, col]
                                destination_piece = self.board[to_row, to_col]
                                self.board[to_row, to_col] = piece
                                self.board[row, col] = 0
                                if not self.check_check(player):
                                    self.board[row, col] = piece
                                    self.board[to_row, to_col] = destination_piece
                                    return False
                                self.board[row, col] = piece
                                self.board[to_row, to_col] = destination_piece
        return True

    def check_check(self, player):
        king_position = None
        for row in range(8):
            for col in range(8):
                if self.board[row, col] == 6 * player:
                    king_position = (row, col)
                    break
            if king_position:
                break

        if not king_position:
            return True  # Return True if the king is not found (considered in checkmate)

        for row_offset in range(-1, 2):
            for col_offset in range(-1, 2):
                if (row_offset, col_offset) != (0, 0):
                    new_position = (king_position[0] + row_offset, king_position[1] + col_offset)
                    if self.is_valid_square(*new_position) and self.is_legal_move(king_position, new_position):
                        return False  # King can move to a safe square, not in check

        return True  # No valid moves for the king, considered in check

    def switch_player(self):
        self.player = -self.player