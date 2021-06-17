import random
from typing import List, Tuple
import plotly.graph_objects as go
from plotly.io import write_image

ALPHA = 0.35
BETA = 0.5
TRANSITION_MATRIX: List[List[float]] = [[1 - ALPHA, ALPHA], [BETA, 1 - BETA]]


def mc_step(curr_state: int, transition_matrix: List[List[float]]) -> int:
    rand = random.random()
    if (curr_state == 1) & (rand < transition_matrix[0][1]):
        return -1
    elif (curr_state == -1) & (rand < transition_matrix[1][0]):
        return 1
    else:
        return curr_state


def markov_chain_simulation(simulation_steps: List[int] = None) -> None:
    if simulation_steps is None:
        simulation_steps = [100, 500, 10 ** 3,
                            5 * 10 ** 4, 10 ** 5, 5 * 10 ** 5, 10 ** 6]

    for steps in simulation_steps:
        curr_step = -1 if random.random() < 0.5 else 1
        sum_plus_1 = 0
        sum_minus_1 = 0
        data_frame: List[Tuple[int, int, int, int]] = []
        for i in range(1, steps + 1):
            curr_step = mc_step(curr_step, TRANSITION_MATRIX)
            if curr_step == 1:
                sum_plus_1 += 1
            if curr_step == -1:
                sum_minus_1 += 1
            data_frame.append((i, curr_step, sum_plus_1, sum_minus_1))
        create_trajectory_plot(data_frame)
        probability_distribution_plot(data_frame)


def create_trajectory_plot(data_frame: List[Tuple[int, int, int, int]]) -> None:
    fig = go.Figure()
    x = list(map(lambda tuple_row: tuple_row[0], data_frame))
    y = list(map(lambda tuple_row: tuple_row[1], data_frame))
    fig.add_trace(go.Line(x=x, y=y))
    fig.update_layout(
        {'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'}
        , width=1000,
        height=800
    )
    fig.update_xaxes(
        linewidth=1, linecolor='black',
        mirror=True, ticks='outside',
        tickwidth=2,
        tickfont=dict({'size': 13}), ticklen=8, gridcolor='white',
        title='', title_font_size=20,
        title_font_color='black', color='black',
    )
    fig.update_yaxes(
        linewidth=1,
        linecolor='black', mirror=True,
        ticks='outside', tickwidth=2,
        ticklen=8, title='',
        title_font_size=20, title_font_color='black',
        color='black', tickfont=dict({'size': 13})
    )
    # write_image(fig, f"trajectory_plot={len(x)}.png")
    fig.show()
    pass


def probability_distribution_plot(data_frame: List[Tuple[int, int, int, int]]) -> None:
    fig = go.Figure()
    x = [i for i in range(len(data_frame))]
    y_p1 = list(map(lambda tuple_row: tuple_row[2] / (tuple_row[2] + tuple_row[3]), data_frame))
    y_p2 = list(map(lambda tuple_row: tuple_row[3] / (tuple_row[2] + tuple_row[3]), data_frame))

    calculated_p1 = 1 / (ALPHA / BETA + 1)
    calculated_p2 = 1 / (BETA / ALPHA + 1)
    fig.add_trace(go.Line(x=[0, len(x)], y=[calculated_p1, calculated_p1], name="expected p1"))
    fig.add_trace(go.Line(x=[0, len(x)], y=[calculated_p2, calculated_p2], name="expected p2"))

    data_frame_p1 = dict({'x': x, 'y': y_p1})
    fig.add_trace(go.Scattergl(
        data_frame_p1, x=x, y=y_p1, mode='markers', name=f"p1",
        marker=dict(size=3, color='brown', symbol='triangle-up-open-dot')
    ))

    data_frame_p2 = dict({'x': x, 'y': y_p2})
    fig.add_trace(go.Scattergl(
        data_frame_p2, x=x, y=y_p2, mode='markers', name=f"p2",
        marker=dict(size=3, color='green', symbol='cross-open-dot')
    ))

    fig.update_layout(
        {'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'}
        , width=1000,
        height=800,
        legend=dict(
            x=0.85,
            y=1,
            traceorder='normal',
            font=dict(size=13, color='black'),
        )
    )
    fig.update_xaxes(
        linewidth=1,
        linecolor='black', mirror=True,
        ticks='outside', tickwidth=2,
        tickfont=dict({'size': 13}), ticklen=8,
        gridcolor='white', title='',
        title_font_size=20, title_font_color='black',
        color='black',
    )
    fig.update_yaxes(
        linewidth=1, linecolor='black',
        mirror=True, ticks='outside',
        tickwidth=2, ticklen=8,
        title='',
        title_font_size=20, title_font_color='black',
        color='black', tickfont=dict({'size': 13})
    )
    # write_image(fig, f"probability_distribution_plotN={len(x)}.png")
    fig.show()

    pass


if __name__ == '__main__':
    markov_chain_simulation()
    pass
