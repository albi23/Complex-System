import random as rand
from functools import reduce
from typing import List
import plotly.graph_objects as go
from plotly.io import write_image
from collections import defaultdict


def opinion_simulation_1(d: float = 0.1, N: int = 1_000) -> None:
    u: float = 0.5
    time_unit = 1
    MCS = 50_000

    opinions: List[float] = [rand.random() for _ in range(N)]
    x: List[int] = []
    y: List[float] = []

    for i in range(MCS):
        # print("\r" + str((float(i + 1) / MCS) * 100), end="%")
        for j in range(time_unit):
            agent_1: int = rand.randint(0, N - 1)
            agent_2: int = rand.randint(0, N - 1)
            while agent_1 == agent_2:
                agent_2 = rand.randint(0, N - 1)
            if abs(opinions[agent_1] - opinions[agent_2]) < d:
                opinions[agent_1] = opinions[agent_1] + u * (opinions[agent_2] - opinions[agent_1])
                opinions[agent_2] = opinions[agent_2] + u * (opinions[agent_1] - opinions[agent_2])
                x.append(i + 1)
                y.append(opinions[agent_1])
                x.append(i + 1)
                y.append(opinions[agent_2])

    peeks_scater_plot(x, y)


def opinion_simulation_2(d: float = 0.15, N: int = 1_000) -> None:
    u: float = 0.5
    time_unit = 1
    MCS = 250

    initial_opinions: List[float] = [rand.random() for _ in range(N)]
    opinions_copy: List[float] = [op for op in initial_opinions]
    x: List[int] = []
    y: List[float] = []

    for i in range(MCS):
        # print("\r" + str((float(i + 1) / MCS) * 100), end="%")
        for j in range(time_unit):
            agent_1: int = rand.randint(0, N - 1)
            agent_2: int = rand.randint(0, N - 1)
            while agent_1 == agent_2:
                agent_2 = rand.randint(0, N - 1)
            if abs(opinions_copy[agent_1] - opinions_copy[agent_2]) < d:
                opinions_copy[agent_1] = opinions_copy[agent_1] + u * (opinions_copy[agent_2] - opinions_copy[agent_1])
                opinions_copy[agent_2] = opinions_copy[agent_2] + u * (opinions_copy[agent_1] - opinions_copy[agent_2])

    final_opinion_vs_initial(initial_opinions, opinions_copy)


def opinion_peeks(d: float = 0.15, N: int = 1_000) -> dict:
    u: float = 0.5
    samples = 250
    MCS = 80_000

    tmp_dict = defaultdict(lambda: 0)
    for i in range(samples):
        initial_opinions: List[float] = [rand.random() for _ in range(N)]
        opinions: List[float] = [op for op in initial_opinions]
        print("\r" + str((float(i + 1) / samples) * 100), end="%")
        for _ in range(MCS):
            agent_1: int = rand.randint(0, N - 1)
            agent_2: int = rand.randint(0, N - 1)
            while agent_1 == agent_2:
                agent_2 = rand.randint(0, N - 1)
            if abs(opinions[agent_1] - opinions[agent_2]) < d:
                opinions[agent_1] = opinions[agent_1] + u * (opinions[agent_2] - opinions[agent_1])
                opinions[agent_2] = opinions[agent_2] + u * (opinions[agent_1] - opinions[agent_2])

        final_opinion_vs_initial(initial_opinions, opinions)
        peeks = count_peeks(N, d, opinions)
        tmp_dict[peeks] = tmp_dict[peeks] + 1
    return {k: v for k, v in tmp_dict.items() if k < 8}


def count_peeks(N: int, d: float, opinions: List[float]) -> int:
    opinions = sorted(opinions)
    clusters = {}
    curr = opinions[0]
    dis = d * 0.5
    p_max = 1 / (d * 0.5)
    min_cluster_size = N / p_max
    min_cluster_size = 50
    i = 0
    while i < len(opinions):
        counter = 1
        while i < len(opinions) and curr + dis > opinions[i]:
            i += 1
            counter += 1
        clusters[curr] = counter
        if i < len(opinions):
            curr = opinions[i]
        i += 1
    len1 = len(list(filter(lambda x: x > min_cluster_size, clusters.values())))
    return len1


def peeks_scater_plot(x, y) -> None:
    df2 = dict({
        'x': x,
        'y': y,
    })
    import plotly.express as px

    fig = px.scatter(df2, x="x", y="y")

    fig.update_traces(marker=dict(size=8, color='black', symbol='diamond-open'), mode='markers', name=f"opinions")
    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'},
                      xaxis_tickformat='d',
                      xaxis=dict(tick0=0,
                                 dtick=5_000),
                      showlegend=True,
                      legend=dict(
                          x=0.92,
                          y=1,
                          traceorder='normal',
                          font=dict(size=13, color='black'),
                      )
                      )
    fig['data'][0]['showlegend'] = True

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     tickfont=dict({'size': 13}))
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     tickfont=dict({'size': 13}))

    # write_image(fig, f"opinions.png")
    fig.show()


def final_opinion_vs_initial(x, y) -> None:
    import plotly.express as px
    df2 = dict({'x': x, 'y': y, })

    fig = go.Figure()
    sc1 = go.Scatter(df2, x=x, y=y, mode='markers', marker=dict(size=8, color='black', symbol='cross-open'))
    fig.add_trace(sc1)

    help_fig = px.scatter(df2, x="x", y="y", trendline="ols")
    x_trend = help_fig["data"][1]['x']
    y_trend = help_fig["data"][1]['y']
    fig.add_trace(go.Line(x=[0, 1], y=[0, 1], line={'dash': 'dash', 'color': 'grey'}))
    fig.add_trace(go.Line(x=[0.1, 1], y=[0, 0.9], line={'dash': 'dash', 'color': 'grey'}))
    fig.add_trace(go.Line(x=[0, 0.9], y=[0.1, 1], line={'dash': 'dash', 'color': 'grey'}))
    # fig.add_trace(go.Line(x=x_trend, y=y_trend, line={'dash': 'dash','color': 'grey'}))

    fig.update_layout(
        {'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'},
        xaxis=dict(tick0=0, dtick=0.2),
        yaxis=dict(tick0=0, dtick=0.2),
    )
    #            ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

    fig.update_xaxes(showgrid=True, showline=True, linewidth=1, linecolor='black', mirror=True,
                     ticks='outside', tickwidth=2,
                     gridcolor='black',
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     tickfont=dict({'size': 13}))
    fig.update_yaxes(showgrid=True, showline=True, gridwidth=0.5, linewidth=1, linecolor='black', mirror=True,
                     ticks='outside', tickwidth=2,
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     gridcolor='black',
                     tickfont=dict({'size': 13}))
    # fig.update_traces(marker=dict(size=8, color='black', symbol='cross-open'), mode='markers')

    # write_image(fig, f"opinions.png")
    fig.show()


def simulation_4() -> None:
    steep: float = 0.05
    histogram: dict = defaultdict(lambda: [[], []])
    while steep <= 0.5:
        steep += 0.02
        res = opinion_peeks(steep)
        for k, v in res.items():
            arr = histogram[k]
            arr[0].append(steep)
            arr[1].append(v)

    fig = go.Figure()

    for k, v in histogram.items():
        fig.add_trace(go.Line(x=histogram[k][0], y=histogram[k][1], name=f"{k} peeks"))

    fig.update_layout(
        {'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'}
    )
    fig.update_xaxes(
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='outside',
        tickwidth=2,
        tickfont=dict({'size': 13}),
        ticklen=8,
        gridcolor='white',
        title='',
        title_font_size=20,
        title_font_color='black',
        color='black',
    )
    fig.update_yaxes(
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='outside',
        tickwidth=2,
        ticklen=8,
        title='',
        title_font_size=20,
        title_font_color='black',
        color='black',
        tickfont=dict({'size': 13})
    )
    fig.show()


def simulation_5(lattice_size=29, d: float = 0.3, u: float = 0.3, mcs_steep=100_000) -> None:
    opinions = [[rand.random() for _ in range(lattice_size)] for _ in range(lattice_size)]
    for i in range(mcs_steep):
        print("\r" + str((float(i + 1) / mcs_steep) * 100), end="%")
        agent_1_x: int = rand.randint(0, lattice_size - 1)
        agent_1_y: int = rand.randint(0, lattice_size - 1)
        neighbor_location: int = rand.randint(0, 3)  # N, S, E, W
        agent_2_cord: tuple
        if neighbor_location == 0:
            agent_2_cord = (lattice_size - 1, agent_1_y) if agent_1_x == 0 else (agent_1_x - 1, agent_1_y)
        elif neighbor_location == 1:
            agent_2_cord = (0, agent_1_y) if agent_1_x == lattice_size - 1 else (agent_1_x + 1, agent_1_y)
        elif neighbor_location == 2:
            agent_2_cord = (agent_1_x, lattice_size - 1) if agent_1_y == 0 else (agent_1_x, agent_1_y - 1)
        else:
            agent_2_cord = (agent_1_x, 0) if agent_1_y == lattice_size - 1 else (agent_1_x, agent_1_y + 1)

        if abs(opinions[agent_1_x][agent_1_y] - opinions[agent_2_cord[0]][agent_2_cord[1]]) < d:
            opinions[agent_1_x][agent_1_y] = opinions[agent_1_x][agent_1_y] + u * (
                    opinions[agent_2_cord[0]][agent_2_cord[1]] - opinions[agent_1_x][agent_1_y])
            opinions[agent_2_cord[0]][agent_2_cord[1]] = opinions[agent_2_cord[0]][agent_2_cord[1]] + u * (
                    opinions[agent_1_x][agent_1_y] - opinions[agent_2_cord[0]][agent_2_cord[1]])

    generate_plot_for_simulation_5(d, opinions)


def generate_plot_for_simulation_5(d, opinions, with_heatmap=True):
    if with_heatmap:
        heat_map_plot(opinions)
    clusters_100_times = {k * 100: v for k, v in get_clusters(d, opinions).items()}
    x = list(clusters_100_times.keys())
    y = list(clusters_100_times.values())
    df2 = dict({'x': x, 'y': y, })
    fig = go.Figure()
    sc1 = go.Scatter(
        df2, x=x, y=y, mode='markers',
        marker=dict(size=8, color='black', symbol='cross-open'),
    )
    fig.add_trace(sc1)
    fig.update_layout(
        {'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'},
        yaxis_range=[0, 600],
        xaxis_range=[0, 100],
    )
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', mirror=True,
                     ticks='outside', tickwidth=2,
                     gridcolor='black',
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     tickfont=dict({'size': 13}))
    fig.update_yaxes(showgrid=False, showline=True, gridwidth=0.5, linewidth=1, linecolor='black', mirror=True,
                     ticks='outside', tickwidth=2,
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     gridcolor='black',
                     tickfont=dict({'size': 13}))
    fig.show()


def heat_map_plot(opinions):
    import plotly.figure_factory as ff
    txt = [["" for _ in range(len(opinions))] for _ in range(len(opinions))]
    tmp = []
    for i in range(len(opinions) - 1, -1, -1):
        tmp.append(opinions[i])
    fig = ff.create_annotated_heatmap(tmp, annotation_text=txt,
                                      # colorscale=['rgb(0,0,0)', 'rgb(219,219,219)'],
                                      colorscale=['rgb(230,230,230)', 'rgb(0,0,0)'],
                                      zmin=0, zmax=1, showscale=False, xgap=3, ygap=3)
    fig.update_xaxes(showline=False, showgrid=False, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=False, showgrid=False, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'}, width=1000,
                      height=1000)
    fig.show()


def get_clusters(d: float, opinions: List[List[float]]) -> dict:
    linear_opinion = sorted(flat(opinions))
    clusters = {}
    curr = linear_opinion[0]
    dis = d * 0.5
    i = 0
    while i < len(linear_opinion):
        counter = 1
        while i < len(linear_opinion) and curr + dis > linear_opinion[i]:
            i += 1
            counter += 1
        clusters[curr] = counter
        if i < len(linear_opinion):
            curr = linear_opinion[i]
        i += 1
    return clusters


def flat(opinions):
    linear_opinion = []
    for row in opinions:
        linear_opinion += row
    return linear_opinion


def simulation_5_triangle(triangle_lattice_size: int = 29, mcs_steep: int = 100_000, u=0.3, d=0.5) -> None:
    if triangle_lattice_size % 2 == 0:
        raise ValueError('triangle_lattice_size should be odd')
    opinions = generate_triangle_lattice(triangle_lattice_size)
    rows_count = int(triangle_lattice_size / 2)
    i = 0
    while i < mcs_steep:
        agent_1_x: int = rand.randint(0, rows_count - 1)
        agent_1_y: int = rand.randint(0, len(opinions[agent_1_x]) - 1)
        neighbor_location: int = rand.randint(0, 3)  # N, S, E, W
        agent_2_cord: tuple
        if neighbor_location == 0:  # N
            if agent_1_x == rows_count and (agent_1_y == 0 or agent_1_y == triangle_lattice_size - 1):  # no neighbor
                continue
            agent_2_cord = (rows_count, agent_1_y + (rows_count - agent_1_x)) \
                if agent_1_x == 0 or agent_1_y == 0 or agent_1_y == len(opinions[agent_1_x]) - 1 \
                else (agent_1_x + 1, agent_1_y - 1)
        elif neighbor_location == 1:  # S
            if agent_1_x == rows_count and (agent_1_y == 0 or agent_1_y == triangle_lattice_size - 1):  # no neighbor
                continue
            agent_2_cord = (abs(rows_count - agent_1_y), 0) if agent_1_x == rows_count else (
                agent_1_x + 1, agent_1_y + 1)
        elif neighbor_location == 2:  # W
            if agent_1_x == 0:
                continue
            agent_2_cord = (agent_1_x, 0) if len(opinions[agent_1_x]) - 1 == agent_1_y else (agent_1_x, agent_1_y + 1)
        else:  # E
            if agent_1_x == 0:
                continue
            agent_2_cord = (agent_1_x, len(opinions[agent_1_x]) - 1) if agent_1_y == 0 else (agent_1_x, agent_1_y - 1)
        print("\r" + str((float(i + 1) / mcs_steep) * 100), end="%")
        i += 1
        if abs(opinions[agent_1_x][agent_1_y] - opinions[agent_2_cord[0]][agent_2_cord[1]]) < d:
            opinions[agent_1_x][agent_1_y] = opinions[agent_1_x][agent_1_y] + u * (
                    opinions[agent_2_cord[0]][agent_2_cord[1]] - opinions[agent_1_x][agent_1_y])
            opinions[agent_2_cord[0]][agent_2_cord[1]] = opinions[agent_2_cord[0]][agent_2_cord[1]] + u * (
                    opinions[agent_1_x][agent_1_y] - opinions[agent_2_cord[0]][agent_2_cord[1]])

    to_square_matrix(opinions, rows_count)
    triangle_heat_map(opinions, rows_count, triangle_lattice_size)
    generate_plot_for_simulation_5(d, opinions, False)
    pass


def triangle_heat_map(opinions, rows_count, triangle_lattice_size):
    import plotly.figure_factory as ff
    tmp = []
    for i in range(len(opinions) - 1, -1, -1):
        tmp.append(opinions[i])
    txt = [["" for _ in range(triangle_lattice_size)] for _ in range(rows_count + 1)]
    fig = ff.create_annotated_heatmap(tmp, annotation_text=txt,
                                      colorscale=[
                                          [0, 'rgb(255,255,255)'],
                                          [0.01, 'rgb(255,255,255)'],
                                          [0.01, 'rgb(0,0,0)'],
                                          [1.0, 'rgb(219,219,219)'],
                                      ],
                                      zmin=0, zmax=1, showscale=False, xgap=3, ygap=3)
    fig.update_xaxes(showline=False, showgrid=False, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=False, showgrid=False, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'}, width=1000,
                      height=800)
    fig.show()


def to_square_matrix(opinions, rows_count):
    for i in range(rows_count):
        to_add = rows_count - i
        for _ in range(to_add):
            opinions[i].insert(0, -1)
        for _ in range(to_add):
            opinions[i].append(-1)


def generate_triangle_lattice(size: int) -> List[List[float]]:
    arr = [[rand.random() for _ in range(size)]]
    while size > 1:
        size = 1 if size == 2 else size - 2
        arr.insert(0, [rand.random() for _ in range(size)])
    return arr


def simulation_6(m: int = 13, N: int = 1000, u=0.5, d: int = 2, mc_steps=10 ** 7) -> None:
    opinions_vectors = [[1 if rand.random() > 0.5 else 0 for _ in range(m)] for _ in range(N)]

    for i in range(mc_steps):
        print("\r" + str((float(i + 1) / mc_steps) * 100), end="%")

        agent_1: int = rand.randint(0, N - 1)
        agent_2: int = rand.randint(0, N - 1)

        if hamming_distance(opinions_vectors[agent_1], opinions_vectors[agent_2]) < d:
            if rand.random() < u:  # agent 1 try to convince agent 2 on differ subject
                convince_process(opinions_vectors[agent_1], opinions_vectors[agent_2], m, u)
            else:
                convince_process(opinions_vectors[agent_2], opinions_vectors[agent_1], m, u)

    # distance from X_0
    y: List[int] = []
    for row in opinions_vectors:
        y.append(sum(row))

    x = [i for i in range(N)]
    df2 = dict({'x': x, 'y': y, })
    fig = go.Figure()
    sc1 = go.Scatter(
        df2, x=x, y=y, mode='markers',
        marker=dict(size=8, color='black', symbol='cross-open'),
    )
    fig.add_trace(sc1)
    fig.update_layout(
        {'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'},
        yaxis_range=[0, 13],
        # xaxis_range=[0, 100],
    )
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', mirror=True,
                     ticks='outside', tickwidth=2,
                     gridcolor='black',
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     tickfont=dict({'size': 13}))
    fig.update_yaxes(showgrid=False, showline=True, gridwidth=0.5, linewidth=1, linecolor='black', mirror=True,
                     ticks='outside', tickwidth=2,
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     gridcolor='black',
                     tickfont=dict({'size': 13}))
    fig.show()


def convince_process(agent_1_opinion: List[int], agent_2_opinion: List[int], m: int, u: float):
    for idx in range(m):
        if agent_1_opinion[idx] != agent_2_opinion[idx] and rand.random() < u:
            agent_2_opinion[idx] = agent_1_opinion[idx]


def hamming_distance(vector1: List[int], vector2: List[int]) -> int:
    count = 0
    for i in range(len(vector1)):
        if vector1[i] != vector2[i]:
            count += 1
    return count


def simulation_7(m: int = 13, N: int = 1000, u=1, d: int = 3, samples=1, time_unit=15_000,
                 plot_steps: int = 1) -> None:
    opinions = [[1 if rand.random() > 0.5 else 0 for _ in range(m)] for _ in range(N)]

    counter = 0
    distance_dict = {i: 0 for i in range(m + 1)}
    for step in range(plot_steps):
        print("\r" + str((float(step + 1) / plot_steps) * 100), end="%")
        for i in range(samples):
            for t in range(time_unit):
                agent_1: int = rand.randint(0, N - 1)
                agent_2: int = rand.randint(0, N - 1)

                if hamming_distance(opinions[agent_1], opinions[agent_2]) < d:
                    if rand.random() < u:  # agent 1 try to convince agent 2 on differ subject
                        convince_process(opinions[agent_1], opinions[agent_2], m, u)
                    else:
                        convince_process(opinions[agent_2], opinions[agent_1], m, u)
        counter += 1
        print("counter = " + str(counter))
        for agent_1 in range(N):
            for agent_2 in range(agent_1 + 1, N):
                distance = (m - hamming_distance(opinions[agent_1], opinions[agent_2]))
                distance_dict[distance] += 1

    y = [val / plot_steps for val in distance_dict.values()]
    x = [val for val in distance_dict.keys()]
    df2 = dict({'x': x, 'y': y, })
    fig = go.Figure()
    sc1 = go.Scatter(df2, x=x, y=y, mode='lines+markers', )
    fig.add_trace(sc1)
    fig.update_layout(
        {'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'},
        yaxis=dict(tick0=0, dtick=20_000),
        xaxis=dict(tick0=0, dtick=2),
        xaxis_range=[0, m],
        yaxis_range=[0, 120_000],
        width=800,
        height=800
    )
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', mirror=True,
                     ticks='outside', tickwidth=2,
                     gridcolor='black',
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     tickfont=dict({'size': 13}))
    fig.update_yaxes(showgrid=False, showline=True, gridwidth=0.5, linewidth=1, linecolor='black', mirror=True,
                     ticks='outside', tickwidth=2,
                     ticklen=8, title='', title_font_size=20, title_font_color='black', color='black',
                     gridcolor='black',
                     tickfont=dict({'size': 13}))
    fig.show()


if __name__ == '__main__':
    # opinion_simulation_1()
    # opinion_simulation_2()
    # simulation_4()
    # simulation_5(lattice_size=29, d=0.3, u=0.3, mcs_steep=100_000)
    # simulation_5(lattice_size=29, d=0.15, u=0.3, mcs_steep=100_000)
    # simulation_5_triangle(mcs_steep=100_000, triangle_lattice_size=29, d=0.3, u=0.3)
    # simulation_5_triangle(mcs_steep=100_000, triangle_lattice_size=57, d=0.3, u=0.3)
    # simulation_6(d=7, mc_steps=1_000_000)
    # simulation_6(m=13, d=8,mc_steps=10_000_000) # figure 8 --> zbiega do 9
    # simulation_6(m=13, d=7,mc_steps=10_000_000) # figure 8 --> zbiega do 8
    # simulation_6(m=13, d=4,mc_steps=10_000_000) # figure 8 --> zbiega do 4
    # simulation_6(m=13, d=3,mc_steps=10_000_000) # figure 8 --> zbiega kilku pików
    # simulation_6(m=13, d=2,mc_steps=10_000_000) # figure 8 --> zbiega do kilkunastu pików
    simulation_7()
    pass
