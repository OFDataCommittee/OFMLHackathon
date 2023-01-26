# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import pickle
import os
from pathlib import Path
import numpy as np
import glob
import pathlib

cwd = Path.cwd()

PATH = pathlib.Path(__file__).parent.resolve()
data_folder = "test_training"

def get_num_episode_workers():
    n_workers = len(glob.glob(os.path.join(PATH,  data_folder, "copy_*")))
    n_episodes = len(glob.glob(os.path.join(PATH,  data_folder, "observations_*.pkl")))
    return n_workers, n_episodes

n_workers, n_episodes = get_num_episode_workers()

def load(case_path: str = PATH, type: str = "observations", episode: int = 1):
    file = os.path.join(case_path, data_folder, f'{type}_{episode}.pkl')
    with open(file, "rb") as pf:
        return  pickle.load(pf)

def get_episodic_mean_reward():
    mean_rews = []
    for episode in range(n_episodes):
        episode_mean_rews = []
        data = load(episode=episode)
        episode_mean_rews = [data[worker]['rewards'].mean() for worker in range(n_workers)]
        rew_sum = 0
        for rew in episode_mean_rews:
            rew_sum += rew.item()
        mean_rews.append(rew_sum / n_workers)
    return mean_rews


rews = get_episodic_mean_reward()

app = Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

app.title = "DRL Dashboard"

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

######## ## the APP
app.layout = html.Div([

    html.Div([
        html.Label("Episode number"),
        dcc.Slider(
            id="episode-slider",
            min=1,
            max=n_episodes,
            value=1,
            step=1
        ),
        html.Label("Worker number"),
        dcc.Slider(
            id="worker-slider",
            min=1,
            max=n_workers,
            value=1,
            step=1,
        ),
        dcc.Graph(
            id="Cd-graph", figure={"layout": {"height": 300},}
        ),
        dcc.Graph(
            id="Cl-graph", figure={"layout": {"height": 300}}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Div([
        dcc.Graph(
            id="rewards-graph",
            figure=px.line(x=range(1, n_episodes + 1), y=rews, height=300, labels={'x': 'Episodes [-]', 'y': 'Mean rewards [-]'})
        )
    ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'middle'})

])

@app.callback(
    Output('Cd-graph', 'figure'),
    Output('Cl-graph', 'figure'),
    Input('episode-slider', 'value'),
    Input('worker-slider', 'value'))
def update_figure(episode, worker):
    # filter data
    data = load(episode=episode - 1)
    cd = data[worker - 1]['cd']
    cl = data[worker - 1]['cl']
    steps = range(len(cd))
    cd_fig = px.line(x=steps, y=cd, height=300, labels={'x': 'Steps [-]', 'y': 'Cd [-]'})
    cl_fig = px.line(x=steps, y=cl, height=300, labels={'x': 'Steps [-]', 'y': 'Cl [-]'})
    
    return cd_fig, cl_fig


if __name__ == '__main__':
    app.run_server(debug=True)
