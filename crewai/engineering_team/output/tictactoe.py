class TicTacToe:
    """
    A class to represent the Tic Tac Toe game for two players (X and O).
    """

    def __init__(self):
        """
        Initializes a new Tic Tac Toe game.
        The game starts with an empty board and 'X's turn.
        """
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.game_status = 'ongoing'

    def start_new_game(self):
        """
        Resets the board and starts a new game.
        """
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.game_status = 'ongoing'

    def make_move(self, row: int, col: int) -> bool:
        """
        Allows the current player to make a move at the specified position.
        :param row: The row index (0-2).
        :param col: The column index (0-2).
        :return: True if the move was successful, False if move was invalid.
        """
        if self.validate_move(row, col):
            self.board[row][col] = self.current_player
            self.check_game_status()
            self.switch_player()
            return True
        return False

    def validate_move(self, row: int, col: int) -> bool:
        """
        Validates if the move made by the player is legal.
        :param row: The row index (0-2).
        :param col: The column index (0-2).
        :return: True if the move is legal, False otherwise.
        """
        if not (0 <= row < 3) or not (0 <= col < 3):
            return False  # Out of bounds
        if self.board[row][col] != ' ':
            return False  # Cell is occupied
        if self.game_status != 'ongoing':
            return False  # Game is over
        return True

    def check_game_status(self):
        """
        Checks and updates the game status (win, draw, ongoing).
        """
        if self.detect_win():
            self.game_status = f"{self.current_player} wins"
        elif self.detect_draw():
            self.game_status = "draw"

    def detect_win(self) -> bool:
        """
        Detects if the current player has won the game.
        :return: True if the player has won, False otherwise.
        """
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if all(self.board[i][j] == self.current_player for j in range(3)):  # Row
                return True
            if all(self.board[j][i] == self.current_player for j in range(3)):  # Column
                return True
        if all(self.board[i][i] == self.current_player for i in range(3)):  # Diagonal
            return True
        if all(self.board[i][2 - i] == self.current_player for i in range(3)):  # Anti-diagonal
            return True
        return False

    def detect_draw(self) -> bool:
        """
        Detects if the game is a draw.
        :return: True if the game is a draw, False otherwise.
        """
        return all(cell != ' ' for row in self.board for cell in row)

    def switch_player(self):
        """
        Switches the current player from 'X' to 'O' or from 'O' to 'X'.
        """
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def get_board(self) -> list:
        """
        Returns the current board state.
        :return: A 3x3 list of strings representing the board.
        """
        return self.board

    def get_current_player(self) -> str:
        """
        Returns whose turn it is.
        :return: 'X' or 'O'.
        """
        return self.current_player

    def get_game_status(self) -> str:
        """
        Returns the current game status.
        :return: 'ongoing', 'X wins', 'O wins', or 'draw'.
        """
        return self.game_status