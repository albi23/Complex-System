import math
import random as rand
from typing import List, Tuple
import plotly.graph_objects as go
from plotly.io import write_image
import plotly.figure_factory as ff

MCS = 230_000
K0 = 30_000
markers = ['circle', 'square', 'cross', 'triangle-up']
colors = ['orange', 'black', 'purple', 'green']


def periodic_boundary_condition_step(i: int, j: int, matrix: List[List[int]], t_star: float):
    delta_energy = 2 * (matrix[i][j]) * (
        neighbor_sum(i, j, matrix))
    if delta_energy < 0 or rand.random() <= math.exp(-delta_energy / t_star):
        matrix[i][j] = -matrix[i][j]


def neighbor_sum(i: int, j: int, matrix: List[List[int]]) -> int:
    matrix_size = len(matrix)
    up: int = matrix[matrix_size - 1][j] if i == 0 else matrix[i - 1][j]
    down: int = matrix[0][j] if i == matrix_size - 1 else matrix[i + 1][j]
    left: int = matrix[i][matrix_size - 1] if j == 0 else matrix[i][j - 1]
    right: int = matrix[i][0] if j == matrix_size - 1 else matrix[i][j + 1]
    return up + down + left + right


def generate_square_matrix(size: int) -> List[List[int]]:
    return [[1 if rand.random() > 0.5 else -1 for _ in range(size)] for _ in range(size)]


def msc_step(matrix_model: List[List[int]], t_star: float) -> Tuple[float, float]:
    lattice_size = len(matrix_model)
    avm = 0.0  # average magnetisation
    avg_energy = 0.0
    avg_squared_energy = 0.0
    for k in range(1, MCS + 1, 1):
        for i in range(lattice_size):
            for j in range(lattice_size):
                periodic_boundary_condition_step(i, j, matrix_model, t_star)
        if k > K0 and k % 100 == 0:
            m = 0  # magnetisation
            for i in range(lattice_size):
                for j in range(lattice_size):
                    m += matrix_model[i][j]
            m = m / (lattice_size * lattice_size)
            avm = avm + abs(m)
            energy = 0  # energy
            for i in range(lattice_size):
                for j in range(lattice_size):
                    energy += (matrix_model[i][j] * neighbor_sum(i, j, matrix_model) * 0.5)
            avg_energy += energy
            avg_squared_energy += energy ^ 2
    avm = avm / 2000
    avg_energy = avg_energy / 2000
    avg_squared_energy = avg_squared_energy / 2000
    heat = 1.0 / ((lattice_size ** 2) * (t_star ** 2)) * (avg_squared_energy - (avg_energy ** 2))
    return avm, heat


def construct_scatter(lattice_size=10, t0=1.5, t_end=3.5, step=0.1, color='', marker='', mcs_index=0):
    print(f"Start for L = {lattice_size}")
    x = []
    y = []
    while t0 <= t_end:
        print(t0)
        result_tuple = msc_step(generate_square_matrix(lattice_size), t0)
        x.append(t0)
        y.append(result_tuple[mcs_index])
        t0 += step
    df = dict({'x': x, 'y': y, })
    return go.Scatter(df, x=x, y=y, mode='markers', name=f"L={lattice_size}",
                      marker=dict(size=5, color=color,
                                  symbol=marker))


def avg_magnetization():
    scaters = []
    idx = 0
    for lattice in [5, 10, 30, 60]:
        scaters.append(construct_scatter(lattice, 1.5, 3.5, 0.01, colors[idx], markers[idx], 0))
        idx += 1
    chart_for_avg_magnetization_over_equilibrium_configurations(scaters)


def avg_heat():
    scaters = []
    idx = 0
    for lattice in [8, 16, 35]:
        scaters.append(construct_scatter(lattice, 1.5, 3.5, 0.01, colors[idx], markers[idx], 0))
        idx += 1
    chart_for_avg_heat_over_equilibrium_configurations(scaters)


def example_configurations_of_spins(t_star: float, lattice_size: int) -> List[List[int]]:
    matrix_model = generate_square_matrix(lattice_size)
    for k in range(MCS):
        for i in range(lattice_size):
            for j in range(lattice_size):
                periodic_boundary_condition_step(i, j, matrix_model, t_star)
    return matrix_model


def create_configurations_of_spins_chart(matrix: List[List[int]], L, t_star) -> None:
    annotation_txt = [["" for _ in range(len(matrix))] for _ in range(len(matrix))]
    labels = [i + 1 for i in range(len(matrix))]
    ordered_data = [matrix[i] for i in range(len(matrix) - 1, -1, -1)]
    fig = ff.create_annotated_heatmap(
        ordered_data, annotation_text=annotation_txt,
        colorscale=['rgb(255,255,51)', 'rgb(1,42,99)'],
        zmin=-1, zmax=1,
        showscale=False, x=labels,
        y=labels[::-1]
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(width=600, height=600)
    # write_image(fig, f"spins_L{L}_T*{t_star}.png")
    fig.show()


def chart_for_avg_magnetization_over_equilibrium_configurations(scatter: list) -> None:
    fig = go.Figure()
    for s in scatter:
        fig.add_trace(s)
    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'}, width=600,
                      height=600)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='reduced temperature T*', title_font_size=20, title_font_color='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='average magnetization <m>', title_font_size=20, title_font_color='black')
    fig.show()


def chart_for_avg_heat_over_equilibrium_configurations(scatter: list) -> None:
    fig = go.Figure()
    for s in scatter:
        fig.add_trace(s)
    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'}, width=600,
                      height=600)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='Temperature T*', title_font_size=20, title_font_color='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='Heat Capacity C/N', title_font_size=20, title_font_color='black')
    fig.show()


def example_configuration_spins_simulation():
    l_data: List[int] = [8, 16, 35]
    t_stars: List[float] = [1.0, 2.26, 10.0]
    for l_size in l_data:
        for t in t_stars:
            spins = example_configurations_of_spins(t_star=t, lattice_size=l_size)
            create_configurations_of_spins_chart(spins, L=l_size, t_star=t)


if __name__ == '__main__':
    example_configuration_spins_simulation()
    avg_magnetization()
    avg_heat()
