import unittest
from tictactoe import TicTacToe

class TestTicTacToe(unittest.TestCase):

    def setUp(self):
        self.game = TicTacToe()

    def test_initial_board(self):
        expected_board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
        self.assertEqual(self.game.get_board(), expected_board)

    def test_initial_player(self):
        self.assertEqual(self.game.get_current_player(), 'X')

    def test_valid_move(self):
        self.assertTrue(self.game.make_move(0, 0))
        self.assertEqual(self.game.get_board()[0][0], 'X')

    def test_invalid_move_out_of_bounds(self):
        self.assertFalse(self.game.make_move(-1, 0))
        self.assertFalse(self.game.make_move(3, 0))

    def test_invalid_move_cell_occupied(self):
        self.game.make_move(0, 0)
        self.assertFalse(self.game.make_move(0, 0))

    def test_switch_player(self):
        self.game.make_move(0, 0)
        self.assertEqual(self.game.get_current_player(), 'O')
        self.game.make_move(1, 1)
        self.assertEqual(self.game.get_current_player(), 'X')

    def test_win_condition_row(self):
        self.game.make_move(0, 0)  # X
        self.game.make_move(1, 0)  # O
        self.game.make_move(0, 1)  # X
        self.game.make_move(1, 1)  # O
        self.game.make_move(0, 2)  # X wins
        self.assertEqual(self.game.get_game_status(), 'X wins')

    def test_win_condition_column(self):
        self.game.make_move(0, 0)  # X
        self.game.make_move(0, 1)  # O
        self.game.make_move(1, 0)  # X
        self.game.make_move(1, 1)  # O
        self.game.make_move(2, 0)  # X wins
        self.assertEqual(self.game.get_game_status(), 'X wins')

    def test_win_condition_diagonal(self):
        self.game.make_move(0, 0)  # X
        self.game.make_move(0, 1)  # O
        self.game.make_move(1, 1)  # X
        self.game.make_move(1, 0)  # O
        self.game.make_move(2, 2)  # X wins
        self.assertEqual(self.game.get_game_status(), 'X wins')

    def test_draw_condition(self):
        moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (1, 1), (2, 0), (2, 2), (2, 1)]
        for idx, (row, col) in enumerate(moves):
            self.game.make_move(row, col)
        self.assertEqual(self.game.get_game_status(), 'draw')

    def test_game_restart(self):
        self.game.make_move(0, 0)
        self.game.start_new_game()
        expected_board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
        self.assertEqual(self.game.get_board(), expected_board)
        self.assertEqual(self.game.get_current_player(), 'X')
        self.assertEqual(self.game.get_game_status(), 'ongoing')


if __name__ == '__main__':
    unittest.main()