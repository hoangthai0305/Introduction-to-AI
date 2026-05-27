from abc import ABC, abstractmethod
from copy import deepcopy
import math
import os
from enum import Enum

class Player(Enum):
    HUMAN = 'X'
    AI = 'O'
    EMPTY = ' '

class GameResult(Enum):
    HUMAN_WIN = 1
    AI_WIN = -1
    DRAW = 0
    ONGOING = None

class GameState(ABC):
    @abstractmethod
    def get_legal_actions(self):
        pass

    @abstractmethod
    def make_move(self, action):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def get_utility(self):
        pass

    @abstractmethod
    def is_maximizing_player(self):
        pass

class WeightedLineHeuristic:
    def __init__(self, board_size, win_length):
        self.board_size = board_size
        self.win_length = win_length
        self.WIN_SCORE = 10**(win_length + 2)
        self.LOSE_SCORE = -10**(win_length + 2)

    def evaluate(self, board):
        total_value = 0
        # Horizontal lines
        for r in range(self.board_size):
            for c in range(self.board_size - self.win_length + 1):
                total_value += self._eval_line_segment(board, r, c, 0, 1)
        # Vertical lines
        for r in range(self.board_size - self.win_length + 1):
            for c in range(self.board_size):
                total_value += self._eval_line_segment(board, r, c, 1, 0)
        # Diagonal (top-left to bottom-right)
        for r in range(self.board_size - self.win_length + 1):
            for c in range(self.board_size - self.win_length + 1):
                total_value += self._eval_line_segment(board, r, c, 1, 1)
        # Diagonal (top-right to bottom-left)
        for r in range(self.board_size - self.win_length + 1):
            for c in range(self.win_length - 1, self.board_size):
                total_value += self._eval_line_segment(board, r, c, 1, -1)
        return total_value

    def _eval_line_segment(self, board, r_start, c_start,
                           delta_row, delta_col):
        x_count = 0
        o_count = 0
        for i in range(self.win_length):
            r, c = r_start + i * delta_row, c_start + i * delta_col
            if board[r][c] == Player.HUMAN.value:
                x_count += 1
            elif board[r][c] == Player.AI.value:
                o_count += 1

        if x_count > 0 and o_count == 0: # Line benefits X (Human)
            return 10**x_count
        elif o_count > 0 and x_count == 0: # Line benefits O (AI)
            return -(10**o_count)
        return 0  # Line is blocked by both players, empty, or not a threat

class AlphaBetaSearch:
    def __init__(self, max_depth = 4):
        self.max_depth = max_depth
        self.nodes_explored = 0

    def search(self, state):
        self.nodes_explored = 0
        best_action = None
        if state.is_maximizing_player():
            _, best_action = self._max_value(state, -math.inf, math.inf, self.max_depth)
        else:
            _, best_action = self._min_value(state, -math.inf, math.inf, self.max_depth)
        return best_action

    def _max_value(self, state, alpha, beta, depth):
        self.nodes_explored += 1
        if depth <= 0 or state.is_terminal():
            return state.get_utility(), None

        max_eval = -math.inf
        best_action = None
        legal_actions = state.get_legal_actions()
        if not legal_actions:
             return state.get_utility(), None

        if best_action is None and legal_actions:
            best_action = legal_actions[0]

        for action in legal_actions:
            new_state = state.make_move(action)
            eval_score, _ = self._min_value(new_state, alpha, beta, depth - 1)
            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_action

    def _min_value(self, state, alpha, beta, depth):
        self.nodes_explored += 1
        if depth <= 0 or state.is_terminal():
            return state.get_utility(), None

        min_eval = math.inf
        best_action = None
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return state.get_utility(), None

        if best_action is None and legal_actions:
            best_action = legal_actions[0]

        for action in legal_actions:
            new_state = state.make_move(action)
            eval_score, _ = self._max_value(new_state, alpha, beta, depth - 1)
            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_action

class Board(GameState):
    def __init__(self, size = 9, win_length = 4,
                 current_player = Player.HUMAN,
                 board_state = None,
                 last_move = None):
        self.size = size
        self.win_length = win_length
        self.current_player = current_player
        self.last_move = last_move
        if board_state is None:
            self.board = [[Player.EMPTY.value for _ in range(size)] for _ in range(size)]
        else:
            self.board = deepcopy(board_state)
        self.heuristic = WeightedLineHeuristic(self.size, self.win_length)

    def _check_for_winner(self, board_to_check):
        # Check rows
        for r in range(self.size):
            for c in range(self.size - self.win_length + 1):
                if all(board_to_check[r][c+i] == Player.HUMAN.value for i in range(self.win_length)): return Player.HUMAN
                if all(board_to_check[r][c+i] == Player.AI.value for i in range(self.win_length)): return Player.AI
        # Check columns
        for c in range(self.size):
            for r in range(self.size - self.win_length + 1):
                if all(board_to_check[r+i][c] == Player.HUMAN.value for i in range(self.win_length)): return Player.HUMAN
                if all(board_to_check[r+i][c] == Player.AI.value for i in range(self.win_length)): return Player.AI
        # Check main diagonals (top-left to bottom-right)
        for r in range(self.size - self.win_length + 1):
            for c in range(self.size - self.win_length + 1):
                if all(board_to_check[r+i][c+i] == Player.HUMAN.value for i in range(self.win_length)): return Player.HUMAN
                if all(board_to_check[r+i][c+i] == Player.AI.value for i in range(self.win_length)): return Player.AI
        # Check anti-diagonals (top-right to bottom-left)
        for r in range(self.size - self.win_length + 1):
            for c in range(self.win_length - 1, self.size):
                if all(board_to_check[r+i][c-i] == Player.HUMAN.value for i in range(self.win_length)): return Player.HUMAN
                if all(board_to_check[r+i][c-i] == Player.AI.value for i in range(self.win_length)): return Player.AI
        return None

    def _is_draw(self, board_to_check): 
        if self._check_for_winner(board_to_check) is not None:
            return False
        for r in range(self.size):
            for c in range(self.size):
                if board_to_check[r][c] == Player.EMPTY.value:
                    return False 
        return True 

    def get_legal_actions(self):
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == Player.EMPTY.value:
                    actions.append((i, j))
        return actions

    def make_move(self, action):
        row, col = action
        if not self.is_valid_move(row, col):
            raise ValueError(f"Invalid move: ({row+1}, {col+1}) on board where player is {self.current_player.value}")
        new_board_state = deepcopy(self.board)
        new_board_state[row][col] = self.current_player.value
        next_player = Player.AI if self.current_player == Player.HUMAN else Player.HUMAN
        return Board(self.size, self.win_length, next_player, new_board_state, action)

    def is_valid_move(self, row, col):
        return (0 <= row < self.size and
                0 <= col < self.size and
                self.board[row][col] == Player.EMPTY.value)

    def is_terminal(self):
        if self._check_for_winner(self.board) is not None:
            return True
        if not self.get_legal_actions(): 
            return True
        return False

    def get_utility(self):
        winner = self._check_for_winner(self.board)
        if winner == Player.HUMAN:
            return self.heuristic.WIN_SCORE
        elif winner == Player.AI:
            return self.heuristic.LOSE_SCORE
        
        if not self.get_legal_actions():
            return 0

        return self.heuristic.evaluate(self.board)

    def is_maximizing_player(self):
        return self.current_player == Player.HUMAN

    def get_game_result(self):
        winner = self._check_for_winner(self.board)
        if winner == Player.HUMAN:
            return GameResult.HUMAN_WIN
        elif winner == Player.AI:
            return GameResult.AI_WIN
        elif not self.get_legal_actions(): 
            return GameResult.DRAW
        else:
            return GameResult.ONGOING

    def display(self):
        lines = []
        label_width = len(str(self.size))
        cell_width = 3
        col_header = " " * (label_width + 1) + " ".join(f"{j:>{cell_width}}" for j in range(1, self.size + 1))
        lines.append(col_header)
        top_border = "  " * label_width + "+---" * self.size + "+"
        lines.append(top_border)
        for i, row_data in enumerate(self.board):
            row_display = []
            for j, cell in enumerate(row_data):
                cell_str = cell if cell != ' ' else ' '
                centered_cell = cell_str.center(cell_width)
                if self.last_move and self.last_move == (i, j):
                    if cell == Player.HUMAN.value:
                        centered_cell = f"\033[91m{centered_cell}\033[0m"
                    elif cell == Player.AI.value:
                        centered_cell = f"\033[94m{centered_cell}\033[0m"
                row_display.append(centered_cell)
            line = f"{i + 1:>{label_width}} |" + "|".join(row_display) + "|"
            lines.append(line)
            if i < self.size - 1:
                separator = "  " * label_width + "+---" * self.size + "+"
                lines.append(separator)
        bottom_border = "  " * label_width + "+---" * self.size + "+"
        lines.append(bottom_border)
        return "\n".join(lines)

class HumanPlayer:
    def __init__(self, board_size):
        self.board_size = board_size

    def get_move(self, board):
        while True:
            try:
                user_input = input("Enter your move (row col): ").strip()
                row, col = map(int, user_input.split())
                row -= 1
                col -= 1
                if board.is_valid_move(row, col):
                    return (row, col)
                else:
                    print("Invalid move! Try again.")
            except (ValueError, IndexError):
                print("Invalid input! Please enter two numbers separated by space.")

class AIPlayer:
    def __init__(self, max_depth = 4):
        self.search_algorithm = AlphaBetaSearch(max_depth)

    def get_move(self, board):
        print("AI is thinking...")
        move = self.search_algorithm.search(board)
        print(f"AI explored {self.search_algorithm.nodes_explored} nodes")
        return move

class TicTacToeGame:
    def __init__(self):
        self.board = None
        self.human_player = None
        self.ai_player = None
        self.human_goes_first = True

    def setup_game(self):
        print("=== 9x9 Tic-Tac-Toe Setup ===")
        while True:
            try:
                size_input = input("Enter board size (default 9): ").strip()
                if not size_input:
                    board_size = 9
                else:
                    board_size = int(size_input)
                if board_size < 3:
                    print("Board size must be at least 3!")
                    continue
                break
            except ValueError:
                print("Please enter a valid number!")

        while True:
            try:
                win_input = input(f"Enter win length (default 4, max {board_size}): ").strip()
                if not win_input:
                    win_length = 4
                else:
                    win_length = int(win_input)
                if win_length < 3 or win_length > board_size:
                    print(f"Win length must be between 3 and {board_size}!")
                    continue
                break
            except ValueError:
                print("Please enter a valid number!")

        while True:
            try:
                depth_input = input("Enter AI search depth (default 1): ").strip()
                if not depth_input:
                    ai_depth = 1
                else:
                    ai_depth = int(depth_input)
                if ai_depth < 1 or ai_depth > 5:
                    print("Depth should be between 1 and 5 for performance")
                    continue
                break
            except ValueError:
                print("Please enter a valid number!")

        while True:
            first_input = input("Do you want to go first? (y/n, default y): ").strip().lower()
            if not first_input or first_input == 'y':
                self.human_goes_first = True
                break
            elif first_input == 'n':
                self.human_goes_first = False
                break
            else:
                print("Please enter 'y' or 'n'!")

        self.board = Board(board_size, win_length)
        self.human_player = HumanPlayer(board_size)
        self.ai_player = AIPlayer(ai_depth)
        self._clear_screen()

    def play_game(self):
        print("=== Game Started ===")
        print("You are X, AI is O")
        print("Enter moves as 'row col' (e.g., '5 5' for center)")
        print()

        while not self.board.is_terminal():
            print(self.board.display())
            print()
            if self.board.current_player == Player.HUMAN:
                print(f"Your turn! (Legal moves: {len(self.board.get_legal_actions())})")
                move = self.human_player.get_move(self.board)
                self.board = self.board.make_move(move)
            else:
                print("AI's turn...")
                move = self.ai_player.get_move(self.board)
                if move is None:
                    print("AI could not find a move. This might indicate a draw or an issue.")
                    break
                print(f"AI plays: {move[0]+1} {move[1]+1}")
                self.board = self.board.make_move(move)
            self._clear_screen()

        print(self.board.display())
        print()
        result = self.board.get_game_result()
        if result == GameResult.HUMAN_WIN:
            print("Congratulations! You won!")
        elif result == GameResult.AI_WIN:
            print("AI wins! Better luck next time!")
        else:
            print("It's a draw! Good game!")

    def run(self):
        try:
            self.setup_game()
            if not self.human_goes_first:
                self.board.current_player = Player.AI
            self.play_game()
        except KeyboardInterrupt:
            print("\n\nGame interrupted. Thanks for playing!")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

    @staticmethod
    def _clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    game = TicTacToeGame()
    game.run()