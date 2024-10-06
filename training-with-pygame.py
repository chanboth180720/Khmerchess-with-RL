class KhmerChessGame:
    WIN_REWARD = 10
    CAPTURE_REWARD = 1
    ILLEGAL_MOVE_PENALTY = -1
    NO_MOVE_PENALTY = -1
    REPLAY_BUFFER_SIZE = 10000
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.9
    
    HONOR_COUNT_LIMIT = 64
    PIECE_HONOR_COUNT_LIMIT = {
        'two_rooks': 8,
        'one_rook': 16,
        'two_bishops': 22,
        'two_knights': 32,
        'one_bishop': 44,
        'one_knight': 64,
        'queen_and_promoted_pawns': 64
    }

    def __init__(self, policy_network, model_path=None):
        self.board = KhmerChessBoard()
        self.move_count = 0
        self.player_rewards = {1: 0, -1: 0}
        self.results = {"Player 1 Wins": 0, "Player -1 Wins": 0, "Draws": 0}
        self.policy_network = policy_network
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.LEARNING_RATE)
        self.reward_history = {"Player 1": [], "Player -1": []}
        self.moves_history = []
        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_SIZE)
        self.chasing_move_count = 0
        self.escaping_move_count = 0
        self.game_lengths = []

        if model_path:
            self.load_policy_network(model_path)

    def load_policy_network(self, model_path):
        try:
            checkpoint = torch.load(model_path)
            if 'model_state_dict' in checkpoint:
                self.policy_network.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.policy_network.load_state_dict(checkpoint['state_dict'])
            else:
                raise KeyError("Checkpoint does not contain 'state_dict' or 'model_state_dict'")
            self.policy_network.eval()
            print(f"Loaded policy network from {model_path}")
        except Exception as e:
            print(f"Error loading policy network: {e}")

    def play_game(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 800))  # Set up the Pygame window
        pygame.display.set_caption("Khmer Chess")
        clock = pygame.time.Clock()
        running = True

        # Reset variables and board
        self.board.reset()  # Reset the game board
        self.chasing_move_count = 0
        self.escaping_move_count = 0
        self.move_count = 0
        state_action_rewards = []

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Check for game over condition
            if self.is_game_over():
                self.display_game_results()  # Display the final results
                running = False
                continue

            state = self.get_state()
            legal_moves = self.board.get_all_legal_moves(self.board.player)
            print(f"Player {self.board.player} legal moves: {legal_moves}")

            if not legal_moves:
                print(f"No legal moves for Player {self.board.player}.")
                self.player_rewards[self.board.player] += self.NO_MOVE_PENALTY
                self.board.switch_player()  # Switch to the other player if no moves
                continue

            if self.board.player == 1:  # Policy Gradient AI (Player 1)
                print(f"Player 1 (Policy Gradient AI) is making a move.")
                action_index = self.select_action(state, legal_moves)  # Use the policy network to select an action
                action = legal_moves[action_index]  # Get the move
                reward = self.take_action(action)  # Take the move and get the reward
                state_action_rewards.append((state, action_index, reward))  # Track state, action, and reward
                self.chasing_move_count += 1  # Chasing player move count

            elif self.board.player == -1:  # Random AI (Player -1)
                print(f"Player 2 (Random AI) is making a move.")
                self.make_random_move()  # Random move for player -1
                # Switch player after random move to allow Player 1 (AI) to play next
                self.escaping_move_count += 1  # Escaping player move count
                self.board.switch_player()

            # Draw the board using Pygame
            self.draw_board(screen)
            pygame.display.flip()
            clock.tick(100)  # Control game speed (FPS)
            
            # Optional: Add a delay after each action to slow things down further
            pygame.time.delay(100)  # Delay of 1000 milliseconds (1 second) after each action

        pygame.quit()  # Quit Pygame when the game loop ends

    def select_action(self, state, legal_moves):
        state_tensor = state.clone().detach()
        action_probs = self.policy_network(state_tensor).squeeze()
        legal_action_probs = torch.zeros(len(legal_moves))
        for idx, move in enumerate(legal_moves):
            from_square, to_square = move
            to_index = to_square[0] * 8 + to_square[1]
            legal_action_probs[idx] = action_probs[to_index]

        legal_action_probs_sum = legal_action_probs.sum()
        if legal_action_probs_sum > 0:
            legal_action_probs /= legal_action_probs_sum

        action_idx = torch.multinomial(legal_action_probs, 1).item()
        return action_idx

    def take_action(self, action):
        player = self.board.player
        from_square, to_square = action
        reward = 0

        if self.board.move_piece(from_square, to_square):
            if self.board.is_opponent_piece(to_square[0], to_square[1], player):
                reward += self.CAPTURE_REWARD
        else:
            reward += self.ILLEGAL_MOVE_PENALTY

        return reward

    def make_random_move(self):
        legal_moves = self.board.get_all_legal_moves(self.board.player)
        print(f"Random AI has legal moves: {legal_moves}")
        
        if legal_moves:
            move = random.choice(legal_moves)
            print(f"Random AI selects move: {move}")
            self.board.move_piece(move[0], move[1])  # Move the piece
            self.board.switch_player()  # Switch player after making the move
        else:
            print("Random AI has no legal moves.")

    def get_state(self):
        state = torch.zeros(12, 8, 8)
        for row in range(8):
            for col in range(8):
                piece = self.board.board[row, col]
                if piece != 0:
                    piece_type = abs(piece)
                    player = 0 if piece > 0 else 1
                    channel = (piece_type - 1) * 2 + player
                    state[channel, row, col] = 1
        state = state.unsqueeze(0)  # Add batch dimension
        return state

    def is_game_over(self):
        # Check if the game is over due to checkmate
        if self.board.check_checkmate(self.board.player):
            winner = -self.board.player
            print("Player", winner, "wins!")
            self.player_rewards[winner] += self.WIN_REWARD
            self.results[f"Player {winner} Wins"] += 1
            return True

        # Check if all pawns are promoted
        all_pawns_promoted = False
        if self.board.player == 1:
            # Player 1 pawns
            all_pawns_promoted = (self.board.board[1:7, :] != 1).all().item()
        else:
            # Player -1 pawns
            all_pawns_promoted = (self.board.board[0:6, :] != -1).all().item()

        if all_pawns_promoted:
            # Start Piece's Honor Counting
            total_pieces = abs(self.board.board).sum().item()
            
            # Define piece type and limit based on remaining pieces
            piece_counts = self.count_pieces()
            limit = self.get_piece_honor_count_limit(piece_counts)
            
            if self.chasing_move_count >= limit:
                print(f"Game drawn due to Piece's Honor Counting limit of {limit} moves.")
                self.results["Draws"] += 1
                return True

        return False
    
    def count_pieces(self):
        # Count the pieces on the board
        piece_counts = {
            'rooks': 0,
            'bishops': 0,
            'knights': 0,
            'queens': 0,
            'promoted_pawns': 0
        }
        
        for piece in self.board.board.flatten():
            if piece == 4 or piece == -4:
                piece_counts['rooks'] += 1
            elif piece == 3 or piece == -3:
                piece_counts['bishops'] += 1
            elif piece == 2 or piece == -2:
                piece_counts['knights'] += 1
            elif piece == 5 or piece == -5:
                piece_counts['queens'] += 1
            elif piece == 1 or piece == -1:
                piece_counts['promoted_pawns'] += 1

        return piece_counts

    def get_piece_honor_count_limit(self, piece_counts):
        if piece_counts['rooks'] == 2:
            return self.PIECE_HONOR_COUNT_LIMIT['two_rooks']
        elif piece_counts['rooks'] == 1:
            return self.PIECE_HONOR_COUNT_LIMIT['one_rook']
        elif piece_counts['bishops'] == 2:
            return self.PIECE_HONOR_COUNT_LIMIT['two_bishops']
        elif piece_counts['knights'] == 2:
            return self.PIECE_HONOR_COUNT_LIMIT['two_knights']
        elif piece_counts['bishops'] == 1:
            return self.PIECE_HONOR_COUNT_LIMIT['one_bishop']
        elif piece_counts['knights'] == 1:
            return self.PIECE_HONOR_COUNT_LIMIT['one_knight']
        elif piece_counts['queens'] > 0 or piece_counts['promoted_pawns'] > 0:
            return self.PIECE_HONOR_COUNT_LIMIT['queen_and_promoted_pawns']
        
        return self.HONOR_COUNT_LIMIT  # Default to the max limit if no specific rule applies

    def draw_board(self, screen):
        colors = [(255, 206, 158), (209, 139, 71)]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(screen, color, pygame.Rect(col * 100, row * 100, 100, 100))
                piece = self.board.board[row, col].item()  # Convert tensor to int
                if piece != 0:
                    piece_image = self.get_piece_image(piece)
                    screen.blit(piece_image, (col * 100, row * 100))

    def get_piece_image(self, piece):
        piece_map = {
            1: "imgs/b_pawn.png", -1: "imgs/w_pawn.png",
            2: "imgs/b_knight.png", -2: "imgs/w_knight.png",
            3: "imgs/b_bishop.png", -3: "imgs/w_bishop.png",
            4: "imgs/b_rook.png", -4: "imgs/w_rook.png",
            5: "imgs/b_queen.png", -5: "imgs/w_queen.png",
            6: "imgs/b_king.png", -6: "imgs/w_king.png",
        }
        return pygame.image.load(piece_map[piece])

    def display_game_results(self):
        print("Game Results:")
        print("Player 1 Reward:", self.player_rewards[1])
        print("Player -1 Reward:", self.player_rewards[-1])
        print("Results:", self.results)

    def draw_board(self, screen):
        colors = [(255, 206, 158), (209, 139, 71)]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(screen, color, pygame.Rect(col * 100, row * 100, 100, 100))
                piece = self.board.board[row, col].item()  # Convert tensor to int
                if piece != 0:
                    piece_image = self.get_piece_image(piece)
                    screen.blit(piece_image, (col * 100, row * 100))