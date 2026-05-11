import gradio as gr
from tictactoe import TicTacToe

game = TicTacToe()


def _ui_outputs():
    board = game.get_board()
    cells = [board[r][c] if board[r][c] != " " else " " for r in range(3) for c in range(3)]
    return (*cells, game.get_current_player(), game.get_game_status())


def make_move(row, col):
    game.make_move(row, col)
    return _ui_outputs()


def reset():
    game.start_new_game()
    return _ui_outputs()


with gr.Blocks(title="Tic-Tac-Toe") as demo:
    gr.Markdown("## Tic-Tac-Toe")

    cell_btns = []
    for i in range(3):
        with gr.Row():
            for j in range(3):
                btn = gr.Button(" ", min_width=80)
                cell_btns.append((i, j, btn))

    player_out = gr.Textbox(label="Current Player", interactive=False)
    status_out = gr.Textbox(label="Game Status", interactive=False)
    reset_btn = gr.Button("Reset Game")

    all_outputs = [b for _, _, b in cell_btns] + [player_out, status_out]

    for i, j, btn in cell_btns:
        btn.click(fn=lambda r=i, c=j: make_move(r, c), outputs=all_outputs)

    reset_btn.click(fn=reset, outputs=all_outputs)
    demo.load(fn=reset, outputs=all_outputs)

if __name__ == "__main__":
    demo.launch()
