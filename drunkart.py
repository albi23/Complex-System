import collections
import random
from typing import List, Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go


def drunkard(steps: int) -> Tuple:
    steps_history: List[int] = []
    start: int = 0
    steps_history.append(start)
    for i in range(steps):
        start += 1 if random.random() < 0.5 else -1
        steps_history.append(start)

    return steps_history, start


def multiple_drunkard(drunkards_count: int, steps: int) -> Dict:
    distance_histogram = collections.defaultdict(int)
    for i in range(drunkards_count):
        distance_histogram[drunkard(steps)[1]] += 1
    sorted_dict = {k: v for k, v in sorted(distance_histogram.items(), key=lambda item: item[0])}
    return sorted_dict


def generate_single_drunkard_plot():
    y_data = drunkard(100)[0]

    range_ = [i for i in range(len(y_data))]
    df = dict({'x': range_, 'y': y_data, })
    fig = go.Figure()
    # fig.add_trace(go.Scatter(df, x=range_, y=y_data, mode='lines+markers', line_color='red',  showlegend=True))
    fig.add_trace(go.Scatter(df, x=range_, y=y_data, mode='lines+markers', line_color='red'))

    # fig = px.scatter(df, x="x", y="y" , labels={'x': 'steps', 'y': 'X'},)
    # fig = px.line(df, x="x", y="y", color="continent", line_group="country", hover_name="country", line_shape="spline", render_mode="svg")
    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)',
                       'paper_bgcolor': 'rgb(255, 255, 255)'}, width=800, height=800)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror='allticks', ticks='inside', tickwidth=2,
                     ticklen=10, title='<b>Steps</b>', title_font_size=20, title_font_color='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror='allticks', ticks='inside', tickwidth=2,
                     ticklen=10, title='<b>X</b>', title_font_size=20, title_font_color='black')
    fig.update_traces(marker_line_color='red', selector=dict(type='scatter'))
    fig.show()


def generate_multiple_drunkard_plot():
    data = multiple_drunkard(30_000, 100)
    x = list(data.keys())
    y = list(data.values())
    df = dict({'x': x, 'y': y})

    fig = go.Figure()
    # fig.add_trace(go.Scatter(df, x=range_, y=y_data, mode='lines+markers', line_color='red',  showlegend=True))
    fig.add_trace(go.Bar(df, x=x, y=y))

    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)',
                       'paper_bgcolor': 'rgb(255, 255, 255)'}, width=800, height=800, title="K=10\nN=20",
                      legend_title_side='top left', margin_t=50)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False, ticks='outside', tickwidth=2,
                     ticklen=10, title='<b>X_n</b>', title_font_size=20, title_font_color='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=False, ticks='outside', tickwidth=2,
                     ticklen=10, title='<b>number of drunkards</b>', title_font_size=20, title_font_color='black')
    fig.update_traces(marker_color='red', marker_line_color='red')
    fig.add_annotation(
        xref="x",
        yref="y",
        # The arrow head will be 25% along the x axis, starting from the left
        x=-40,
        # The arrow head will be 40% along the y axis, starting from the bottom
        y=2500,
        showarrow=False,
        text="K= \nN=",
        # arrowhead=2,
    )
    # fig.update_traces(marker_line_color='red', selector=dict(type='scatter'))
    fig.show()


if __name__ == '__main__':
    generate_multiple_drunkard_plot()
    print()
