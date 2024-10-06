class KhmerChessGame:
    WIN_REWARD = 10
    CAPTURE_REWARD = 1
    ILLEGAL_MOVE_PENALTY = -1
    NO_MOVE_PENALTY = -1
    REPLAY_BUFFER_SIZE = 10000
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.9  # Define the discount factor
    
    # Define move limits for different situations
    HONOR_COUNT_LIMIT = 64  # Max moves for Board's Honor Counting
    PIECE_HONOR_COUNT_LIMIT = {
        'two_rooks': 8,
        'one_rook': 16,
        'two_bishops': 22,
        'two_knights': 32,
        'one_bishop': 44,
        'one_knight': 64,
        'queen_and_promoted_pawns': 64
    }

    def __init__(self):
        self.board = KhmerChessBoard()
        self.move_count = 0
        self.player_rewards = {1: 0, -1: 0}
        self.results = {"Player 1 Wins": 0, "Player -1 Wins": 0, "Draws": 0}
        self.policy_network = AdvancedPolicyCNN(input_channels=12, output_size=64)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.LEARNING_RATE)
        self.reward_history = {"Player 1": [], "Player -1": []}
        self.moves_history = []
        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_SIZE)
        
        # Initialize move counters
        self.chasing_move_count = 0
        self.escaping_move_count = 0
        
        # Track game lengths
        self.game_lengths = []

    def play_game(self):
        state_action_rewards = []
        self.chasing_move_count = 0
        self.escaping_move_count = 0
        self.move_count = 0

        while not self.is_game_over():
            state = self.get_state()
            legal_moves = self.board.get_all_legal_moves(self.board.player)

            if not legal_moves:
                self.player_rewards[self.board.player] += self.NO_MOVE_PENALTY
                print("No legal moves available. Player", self.board.player, "loses 1 reward point.")
                self.board.switch_player()
                continue

            if self.board.player == 1:
                action_index = self.select_action(state, legal_moves)
                action = legal_moves[action_index]
                reward = self.take_action(action)
                state_action_rewards.append((state, action_index, reward))
                self.chasing_move_count += 1  # Increase move count for chasing player
            else:
                self.make_random_move()
                self.escaping_move_count += 1  # Increase move count for escaping player
            
            self.move_count += 1  # Increase total move count

        # Apply discount factor to rewards
        discounted_rewards = self.discount_rewards([sar[2] for sar in state_action_rewards])
        state_action_rewards = [(sar[0], sar[1], dr) for sar, dr in zip(state_action_rewards, discounted_rewards)]

        self.replay_buffer.extend(state_action_rewards)
        
        if self.board.player == 1:
            self.update_policy()
            
        self.reward_history["Player 1"].append(self.player_rewards[1])
        self.reward_history["Player -1"].append(self.player_rewards[-1])

        # Record the length of the game
        self.game_lengths.append(self.move_count)
        
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
            self.move_count += 1
        else:
            reward += self.ILLEGAL_MOVE_PENALTY

        return reward

    def update_policy(self):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return

        batch = random.sample(self.replay_buffer, self.BATCH_SIZE)
        rewards = []
        log_probs = []

        for (state, action_index, reward) in batch:
            state_tensor = state.clone().detach()
            action_probs = self.policy_network(state_tensor).squeeze()
            selected_action_prob = action_probs[action_index]
            log_probs.append(torch.log(selected_action_prob))
            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        if policy_loss:
            self.optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()

    def make_random_move(self):
        player = self.board.player
        legal_moves = self.board.get_all_legal_moves(player)
        if not legal_moves:
            self.player_rewards[player] += self.NO_MOVE_PENALTY
            print("No legal moves available. Player", player, "loses 1 reward point.")
            self.board.switch_player()
            return

        move = random.choice(legal_moves)
        self.move_piece_and_show(move)

    def move_piece_and_show(self, move):
        player = self.board.player
        from_square, to_square = move
        if self.board.move_piece(from_square, to_square):
            self.move_count += 1
            piece_type = abs(self.board.board[to_square[0], to_square[1]])
            piece_name = self.get_piece_name(piece_type)
        else:
            print("Invalid move")

    def get_piece_name(self, piece_type):
        piece_names = {1: 'Pawn', 2: 'Knight', 3: 'Bishop', 4: 'Rook', 5: 'Queen', 6: 'King'}
        return piece_names.get(piece_type.item(), "Unknown")

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
        if self.board.check_checkmate(self.board.player):
            winner = -self.board.player
            print("Player", winner, "wins!")
            self.player_rewards[winner] += self.WIN_REWARD
            self.results[f"Player {winner} Wins"] += 1
            return True

        # Honor Count Rule Check
        player_pieces = (self.board.board > 0).sum().item() if self.board.player == 1 else (self.board.board < 0).sum().item()
        if player_pieces <= 3:  # Board's Honor Counting
            if self.chasing_move_count >= self.HONOR_COUNT_LIMIT:
                print(f"Game drawn due to Board's Honor Counting limit of {self.HONOR_COUNT_LIMIT} moves.")
                self.results["Draws"] += 1
                return True
        
        # Piece Honor Count Rule Check
        if not any(self.board.board[1, :] == 1) and not any(self.board.board[6, :] == -1):  # No unpromoted pawns
            # Activate Piece's Honor Counting
            piece_type_count = self.board.board.abs().sum().item() - 1
            limit = self.get_piece_honor_count_limit()
            if self.escaping_move_count >= limit:
                print(f"Game drawn due to Piece's Honor Counting limit of {limit} moves.")
                self.results["Draws"] += 1
                return True

        return False
    
    def get_piece_honor_count_limit(self):
        # Count the pieces on the board
        piece_counts = {
            'rooks': 0,
            'bishops': 0,
            'knights': 0,
            'queens': 0,
            'promoted_pawns': 0
        }
        
        for piece in self.board.board.flatten():
            if piece == 3 or piece == -3:
                piece_counts['rooks'] += 1
            elif piece == 2 or piece == -2:
                piece_counts['bishops'] += 1
            elif piece == 1 or piece == -1:
                piece_counts['knights'] += 1
            elif piece == 5 or piece == -5:
                piece_counts['queens'] += 1
            elif piece == 6 or piece == -6:
                piece_counts['promoted_pawns'] += 1

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

    def discount_rewards(self, rewards):
        """Apply discount factor to rewards."""
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.DISCOUNT_FACTOR * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards
    
    def display_move_count(self):
        print("Move count:", self.move_count)