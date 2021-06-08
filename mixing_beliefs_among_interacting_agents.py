"""

|x - x'| < d

x = x + u * (x' - x)
x' = x' + u * (x - x')

u is the convergence parameter taken between 0 and 0.5

openness to discussion, here represented by threshold d
"""
import random as rand
from typing import List
import plotly.graph_objects as go
from plotly.io import write_image


def opinion_simulation_1(d: float = 0.1, N: int = 1_000) -> None:
    u: float = 0.5
    time_unit = 1
    MCS = 50_000

    opinions: List[float] = [rand.random() for _ in range(N)]
    # x = [0 for _ in opinions]
    # y = [op for op in opinions]
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

    create_charts(x, y)


def opinion_simulation_2(d: float = 0.05, N: int = 1_000) -> None:
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

    create_charts2(initial_opinions, opinions_copy)


def create_charts(x, y) -> None:
    # fig = go.Figure()
    df2 = dict({
        'x': x,
        'y': y,
    })
    import plotly.express as px

    fig = px.scatter(df2, x="x", y="y")

    fig.update_traces(marker=dict(size=8, color='black', symbol='diamond-open'), mode='markers', name=f"opinions")
    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'},
                      # margin=dict(l=0, t=0, r=0, b=0),
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


def create_charts2(x, y) -> None:
    import plotly.express as px
    df2 = dict({'x': x, 'y': y, })

    fig = go.Figure()
    sc1 = go.Scatter(df2, x=x, y=y, mode='markers', marker=dict(size=8, color='black', symbol='cross-open'))
    fig.add_trace(sc1)

    help_fig = px.scatter(df2, x="x", y="y", trendline="ols")
    x_trend = help_fig["data"][1]['x']
    y_trend = help_fig["data"][1]['y']
    fig.add_trace(go.Line(x=[0,1], y=[0, 1], line={'dash': 'dash','color': 'grey'}))
    fig.add_trace(go.Line(x=[0.1,1], y=[0,0.9], line={'dash': 'dash','color': 'grey'}))
    fig.add_trace(go.Line(x=[0,0.9], y=[0.1,1], line={'dash': 'dash','color': 'grey'}))
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


if __name__ == '__main__':
    # opinion_simulation_1()
    opinion_simulation_2()

    #     import plotly.express as px
    #
    #     df = px.data.iris()  # iris is a pandas DataFrame
    #     width_ = df['sepal_width']
    #     l = list(width_)
    #     l.sort()
    #     print(l)
    #     values = list(df['sepal_length'])
    #     values.sort()
    #     print(values)
    #     print(values)
    #     df2 = dict({'x': [2.0, 2.2, 2.2, 2.2, 2.3, 2.3, 2.3, 2.3, 2.4, 2.4, 2.4, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.6, 2.6, 2.6, 2.6, 2.6, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.6, 3.6, 3.6, 3.7, 3.7, 3.7, 3.8, 3.8, 3.8, 3.8, 3.8, 3.8, 3.9, 3.9, 4.0, 4.1, 4.2, 4.4],
    #           'y': [4.3, 4.4, 4.4, 4.4, 4.5, 4.6, 4.6, 4.6, 4.6, 4.7, 4.7, 4.8, 4.8, 4.8, 4.8, 4.8, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.2, 5.2, 5.2, 5.2, 5.3, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.9, 5.9, 5.9, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.2, 6.2, 6.2, 6.2, 6.3, 6.3, 6.3, 6.3, 6.3, 6.3, 6.3, 6.3, 6.3, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.5, 6.5, 6.5, 6.5, 6.5, 6.6, 6.6, 6.7, 6.7, 6.7, 6.7, 6.7, 6.7, 6.7, 6.7, 6.8, 6.8, 6.8, 6.9, 6.9, 6.9, 6.9, 7.0, 7.1, 7.2, 7.2, 7.2, 7.3, 7.4, 7.6, 7.7, 7.7, 7.7, 7.7, 7.9]
    # })
    #     fig = px.scatter(df2, x="x", y="y")
    #     fig.show()
    pass
