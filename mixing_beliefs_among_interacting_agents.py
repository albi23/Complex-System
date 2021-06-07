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


def opinion_simulation_1(d: float = 0.5, N: int=2000) -> None:
    u: float = 0.5
    time_unit = 1
    MCS = 50_000

    opinions: List[float] = [rand.random() for _ in range(N)]
    # x = [0 for _ in opinions]
    # y = [op for op in opinions]
    x: List[int] = []
    y: List[float] = []

    for i in range(MCS):
        print("\r" + str((float(i + 1) / MCS) * 100), end="%")
        for j in range(time_unit):
            agent_1: int = rand.randint(0, N - 1)
            agent_2: int = rand.randint(0, N - 1)
            while agent_1 == agent_2:
                agent_2 = rand.randint(0, N - 1)
            if abs(opinions[agent_1] - opinions[agent_2]) < d:
                opinions[agent_1] = opinions[agent_1] + u * (opinions[agent_2] - opinions[agent_1])
                opinions[agent_2] = opinions[agent_2] + u * (opinions[agent_1] - opinions[agent_2])
                x.append(i + 1)
                x.append(i + 1)
                y.append(opinions[agent_1])
                y.append(opinions[agent_2])

        # s = set(opinions)
        # if i in [100,200,300,400,500]:
        #     print("\r" + str(len(s)))
        # s = set(opinions)
        # for op in s:
        #     x.append(i+1)
        #     y.append(op)
    create_charts(x, y)


def create_charts(x, y) -> None:
    # fig = go.Figure()
    df2 = dict({
        'x': x,
        'y': y,
    })
    import plotly.express as px

    fig = px.scatter(df2, x="x", y="y")
    fig.update_traces(marker=dict(size=8, color='black', symbol='diamond-open'))
    # fig.show()

    # for i in range(1,4,1):
    # fig.add_trace(go.Scatter(df, x='x',y='y',mode='markers', name=f"L=", x0=3,
    #                          marker=dict(size=5, color='black',
    #                                      symbol='octagon-open-dot')))
    # fig.add_trace(go.Scatter(df,x='x1',y='y1', mode='markers', name=f"L=", x0=3,
    #                          marker=dict(size=5, color='black',
    #                                      symbol='octagon-open-dot')))
    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'}
                      # ,width=1200,
                      # height=858
                      )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='x_title', title_font_size=20, title_font_color='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='ytitle', title_font_size=20, title_font_color='black')

    # write_image(fig, f"opinions.png")
    fig.show()


if __name__ == '__main__':
    opinion_simulation_1()

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
