import collections
import math
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
                       'paper_bgcolor': 'rgb(255, 255, 255)'}, width=600, height=600)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror='allticks', ticks='inside', tickwidth=2,
                     ticklen=10, title='<b>Steps</b>', title_font_size=20, title_font_color='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror='allticks', ticks='inside', tickwidth=2,
                     ticklen=10, title='<b>X</b>', title_font_size=20, title_font_color='black')
    fig.update_traces(marker_line_color='red', selector=dict(type='scatter'))
    fig.show()


def generate_multiple_drunkard_plot(drunkards=30_000, steps=100):
    data = multiple_drunkard(drunkards, steps)
    x = list(data.keys())
    y = list(data.values())
    df = dict({'x': x, 'y': y})

    fig = go.Figure()
    # fig.add_trace(go.Scatter(df, x=range_, y=y_data, mode='lines+markers', line_color='red',  showlegend=True))
    fig.add_trace(go.Bar(df, x=x, y=y))

    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'},
                      width=600, height=600, title=f"K={drunkards} N={steps}",
                      legend_title_side='top left', margin_t=50)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False, ticks='outside', tickwidth=2,
                     ticklen=10, title='<b>X_n</b>', title_font_size=20, title_font_color='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=False, ticks='outside', tickwidth=2,
                     ticklen=10, title='<b>number of drunkards</b>', title_font_size=20, title_font_color='black')
    fig.update_traces(marker_color='red', marker_line_color='red')
    # fig.update_traces(marker_line_color='red', selector=dict(type='scatter'))
    fig.show()


def standard_deviation(K=10_000, steps=1_000) -> None:
    x_val = []
    y_val = []
    for step in range(1, steps + 1):
        square_path: int = 0
        current_sum_pos: int = 0
        for n_drunkard in range(1, K + 1):
            x = 0
            for i in range(step):
                x += 1 if random.random() < 0.5 else -1
            square_path = square_path + x ** 2
            current_sum_pos = current_sum_pos + x
        square_path /= K
        current_sum_pos /= K
        x_val.append(math.log(step, 10))
        y_val.append(math.log(math.sqrt((square_path - pow(current_sum_pos, 2))), 10))
        print("\r" + str(step), end="")

    df = dict({'x': x_val, 'y': y_val})
    fig = px.scatter(df, x="x", y="y", trendline="ols")
    fig.update_layout({'plot_bgcolor': 'rgb(255, 255, 255)', 'paper_bgcolor': 'rgb(255, 255, 255)'}, width=600,
                      height=600, title=f"K = {K}, N = {steps}", title_x=0.5, margin_t=30)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='log(σ)', title_font_size=20, title_font_color='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', tickwidth=2,
                     ticklen=8, title='log(n)', title_font_size=20, title_font_color='black')
    fig.update_traces(marker_line_color='red', selector=dict(type='scatter'))
    fig.data[1].line.color = 'red'
    fig.show()

    results = px.get_trendline_results(fig)
    print(results.iloc[0]["px_fit_results"].params)
    results = results.iloc[0]["px_fit_results"].summary()
    print(results)


if __name__ == '__main__':
    # generate_single_drunkard_plot()
    # generate_multiple_drunkard_plot()
    standard_deviation()
