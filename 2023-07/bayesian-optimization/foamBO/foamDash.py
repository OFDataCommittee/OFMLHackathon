#!/usr/bin/env python3

""" Perform Insitu visualization of important metrics

This script defines functions to create Pandas dataframes out
of current optimization (or parameter variation) state.

The current principle is to chug all info through CSV files.
Seperate visualization toolkits (rg. Dash) can pick up these files.

IO operations are not intensive, this should be enough
"""

import hydra, logging
import subprocess as sb
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from scipy import stats

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots

from core import process_input_command
from ax.service.scheduler import Scheduler
from ax.storage.json_store.save import save_experiment

log = logging.getLogger(__name__)

app = Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])

def data_from_experiment(scheduler: Scheduler):
    # Trial Parameters with corresponding objective values
    cfg = scheduler.experiment.runner.cfg
    params_df = pd.DataFrame()
    exp_df = scheduler.experiment.fetch_data().df
    if "trial_index" in exp_df.columns:
        exp_df = exp_df.set_index(["trial_index", "metric_name"]).unstack(level=1)["mean"]
        trials = scheduler.experiment.get_trials_by_indices(range(exp_df.shape[0]))
        for tr in trials:
            params_df = pd.concat([params_df,
                pd.DataFrame({
                    **tr.arm.parameters,
                    **tr._properties,
                    "GenerationModel": scheduler.generation_strategy.model._model_key},
                    index=[tr.index])])
        df = pd.merge(exp_df, params_df, left_index=True, right_index=True)
        df.index.name="trial_index"
        df.to_csv(f"{cfg.problem.name}_report.csv")
    save_experiment(scheduler.experiment, f"{cfg.problem.name}_experiment.json")

@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def dash_main(cfg : DictConfig):
    app.title = cfg.problem.name
    @app.callback(Output('live-update-graph', 'figure'),
                  Input('interval-component', 'n_intervals'))
    def update_graph(fig):
        data = pd.DataFrame()
        try:
            data = pd.read_csv(f"{cfg.problem.name}_report.csv")
        except:
            log.warn("Could not visualize current state")
            return fig
        nrows = len(cfg.problem.objectives.keys())
        fig = make_subplots(rows=nrows, cols=1)
        i=1
        for key, _ in cfg.problem.objectives.items():
            print(np.abs(stats.zscore(data[key])))
            df = data[(np.abs(stats.zscore(data[key])) < 1)]
            ifig = px.scatter(df, x=df.index, y=key, hover_name=key, hover_data=df.columns)
            fig.add_trace(
                ifig['data'][0],
                row=i, col=1
            )
            fig['layout']['xaxis{}'.format(i)]['title']='trial_index'
            fig['layout']['yaxis{}'.format(i)]['title']=key
            i += 1
        return fig

    @app.callback(Output('images', 'children'),
                  Input('interval-component', 'n_intervals'))
    def update_images(children):
        data = pd.DataFrame()
        try:
            data = pd.read_csv(f"{cfg.problem.name}_report.csv")
        except:
            log.warn("Could not visualize current state")
            return []
        df = data.tail(cfg.visualize.n_figures)
        figure_uris = []
        for _, row in df.iterrows():
            case = OmegaConf.create({"name": cfg.meta.clone_destination+row["casename"]})
            image_uri = sb.check_output(list(process_input_command(cfg.visualize.figure_generator,
                case)), cwd=case.name, stderr=sb.PIPE)
            figure_uris.append({ **row.to_dict(),
                "image": image_uri.decode("utf-8").strip(' ').replace('\"', '').replace('\\n', '')})
        return [
            html.Div(style={'width': f'{100/cfg.visualize.n_figures}%', 'float': 'left'},
                children=[
                html.Img(src=uri["image"], width='100%', style={'margin':'10px'}),
                html.Div(children=[
                    html.P(children=elm)
                    for elm in OmegaConf.to_yaml(OmegaConf.create(uri)).splitlines()
                    ])
                ])
        for uri in figure_uris ]

    updates = dcc.Interval(
            id='interval-component',
            interval=float(cfg.visualize.update_interval)*1000, # in milliseconds
            n_intervals=0,
    )

    app.layout = html.Div(children=[
        updates,
        html.H1(children=f'Optimization for {cfg.problem.name}',
                style={'text-align':'center', 'padding':'20px'}),
        html.H2(children=f'Optimization Metrics',
                style={'padding':'5px'}),
        dcc.Graph(id='live-update-graph'),
        html.H2(children=f'Insight into latest trials',
                style={'padding':'5px'}),
        html.Div(id='images', style={'padding':'10px'}),
        html.H2(children=f'Your configuration',
                style={'padding':'5px'}),
        html.Div(children=[
            html.Code(children=OmegaConf.to_yaml(cfg), style={'white-space': 'pre-wrap'})
            ],
            style = {'padding': '20px', 'margin': '10px'}
        )
    ])
    app.run_server(debug=False, port=int(cfg.visualize.port), host=cfg.visualize.host)

if __name__ == '__main__':
    dash_main()
