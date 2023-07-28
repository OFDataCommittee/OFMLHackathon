#!/usr/bin/env python3

""" Perform multi-objective optimization on OpenFOAM cases using FullyBayesianMOO if possible

This script defines functions to perform multi-objective optimization on OpenFOAM
cases given a YAML/JSON config file (Supported through Hydra, default: config.yaml).

We use the Adaptive Experimentation Platform for optimization, PyFOAM for parameter substitution
and Hydra for 0-code configuration.

Output: CSV data for experiment trials

Things to improve:
- Optimization restart? Maybe from JSON file as a start.
- Dependent parameters.

Notes:
- You can also use a single objective
"""

import hydra, logging, json
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from ax.service.ax_client import AxClient, MultiObjective
from ax.service.scheduler import ObjectiveThreshold, Scheduler, SchedulerOptions
from ax.core import OptimizationConfig, Experiment, Objective, MultiObjectiveOptimizationConfig
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.feature_importances import plot_feature_importance_by_feature
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.generation_node import GenerationStep

from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy

from core import *
from foamDash import data_from_experiment

log = logging.getLogger(__name__)

def _find_logger_basefilename(logger):
    """Finds the logger base filename(s) currently there is only one
    """
    log_file = None
    parent = logger.__dict__['parent']
    if parent.__class__.__name__ == 'RootLogger':
        # this is where the file name lives
        for h in logger.__dict__['handlers']:
            if h.__class__.__name__ == 'TimedRotatingFileHandler':
                log_file = h.baseFilename
    else:
        log_file = _find_logger_basefilename(parent)

    return log_file 

# Straight from Ax docs, not well tested!
supported_models = {
    "ALEBO": Models.ALEBO,
    "ALEBO_INITIALIZER": Models.ALEBO_INITIALIZER,
    "BOTORCH": Models.BOTORCH,
    "BOTORCH_MODULAR": Models.BOTORCH_MODULAR,
    "BO_MIXED": Models.BO_MIXED,
    "CONTEXT_SACBO": Models.CONTEXT_SACBO,
    "EMPIRICAL_BAYES_THOMPSON": Models.EMPIRICAL_BAYES_THOMPSON,
    "FACTORIAL": Models.FACTORIAL,
    "FULLYBAYESIAN": Models.FULLYBAYESIAN,
    "FULLYBAYESIANMOO": Models.FULLYBAYESIANMOO,
    "FULLYBAYESIANMOO_MTGP": Models.FULLYBAYESIANMOO_MTGP,
    "FULLYBAYESIAN_MTGP": Models.FULLYBAYESIAN_MTGP,
    "GPEI": Models.GPEI,
    "GPKG": Models.GPKG,
    "GPMES": Models.GPMES,
    "MOO": Models.MOO,
    "MOO_MODULAR": Models.MOO_MODULAR,
    "SOBOL": Models.SOBOL,
    "ST_MTGP": Models.ST_MTGP,
    "ST_MTGP_NEHVI": Models.ST_MTGP_NEHVI,
    "THOMPSON": Models.THOMPSON,
    "UNIFORM": Models.UNIFORM,
}

@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def exp_main(cfg : DictConfig) -> None:
    log.info("============= Configuration =============")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info("=========================================")
    search_space = gen_search_space(cfg.problem)
    metrics = [HPCJobMetric(name=key, cfg=cfg) for key, _ in cfg.problem.objectives.items()]
    objectives=[Objective(metric=m, minimize=item.minimize) for m, (_, item) in zip(metrics, cfg.problem.objectives.items())]
    thresholds=[ObjectiveThreshold(metric=m, bound=float(item.threshold), relative=False) for m, (_, item) in zip(metrics, cfg.problem.objectives.items())]
    ax_client = AxClient(verbose_logging=False)
    optimization_config = MultiObjectiveOptimizationConfig(objective=MultiObjective(objectives), objective_thresholds=thresholds) \
            if len(objectives) > 1 else OptimizationConfig(objectives[0])
    exp = Experiment(
        name=f"{cfg.problem.name}_experiment",
        search_space=ax_client.make_search_space(parameters=search_space, parameter_constraints=[]),
        optimization_config=optimization_config,
        runner=HPCJobRunner(cfg=cfg),
        is_test=False,
    )
    gs = choose_generation_strategy(
        search_space=exp.search_space, 
        no_winsorization=True,
        num_trials=cfg.meta.n_trials,
        max_parallelism_cap=cfg.meta.n_parallel_trials,
        use_saasbo=cfg.meta.use_saasbo,
    ) if cfg.problem.models == 'auto' else GenerationStrategy([
            GenerationStep(
                model=supported_models[mk],
                num_trials=cfg.problem.models[mk],
                max_parallelism=cfg.meta.n_parallel_trials
            )
        for mk in cfg.problem.models])
    log.info(f'Generation Strategy: {gs}')
    scheduler = Scheduler(
        experiment=exp,
        generation_strategy=gs,
        options=SchedulerOptions(
            log_filepath=log.manager.root.handlers[1].baseFilename,
            ttl_seconds_for_trials=cfg.meta.trial_ttl
                if "trial_ttl" in cfg.meta.keys() else None,
            init_seconds_between_polls=cfg.meta.init_poll_wait
                if "init_poll_wait" in cfg.meta.keys() else 1,
            seconds_between_polls_backoff_factor=cfg.meta.poll_factor
                if "poll_factor" in cfg.meta.keys() else 1.5,
            timeout_hours=cfg.meta.timeout_hours
                if "timeout_hours" in cfg.meta.keys() else None,
            global_stopping_strategy=ImprovementGlobalStoppingStrategy(
                min_trials=int(cfg.meta.stopping_strategy.min_trials),
                window_size=cfg.meta.stopping_strategy.window_size,
                improvement_bar=cfg.meta.stopping_strategy.improvement_bar,
            ),
        ),
    )
    # This continuously writes CSV data and saves the experiment
    scheduler.run_n_trials(max_trials=cfg.meta.n_trials,
        ignore_global_stopping_strategy=False if cfg.problem.models == 'auto' else True,
        idle_callback=data_from_experiment)

    if cfg.problem.type == "parameter_variation":
        return

    if not cfg.problem.type in ["parameter_variation", "optimization"]:
        log.warning(f"Problem type not parameter_variation, or optimization. Considering optimiztion...")

    # Get some summary
    scheduler.summarize_final_result()

    if len(objectives) == 1:
        # Single-Objective optimization
        log.info("==== Best Parameter Set ===")
        best_params = scheduler.get_best_parameters()
        log.info(OmegaConf.to_yaml(best_params[0]))
        log.info(best_params[1])
        best_trial = scheduler.get_best_trial()
        log.info(f"==== Best trial was {best_trial[0]} ===")
        log.info(OmegaConf.to_yaml(best_trial[1]))
        log.info(best_trial[2])
        
    else:
        try:
            # Plot Pareto frontier
            log.info("==== Pareto optimal paramters: ===")
            pareto_params = scheduler.get_pareto_optimal_parameters()
            log.info(pareto_params)
            log.info(json.dumps(pareto_params, indent=4))
            log.info("==================================")

            metric_names = [e.metric.name for e in objectives]
            sobol_frontier = compute_posterior_pareto_frontier(
                experiment=exp,
                data=exp.fetch_data(),
                primary_objective=objectives[1].metric,
                secondary_objective=objectives[0].metric,
                absolute_metrics=metric_names,
                num_points=int(cfg.meta.n_pareto_points),
            )
            plot_frontier(sobol_frontier, f"{cfg.problem.name}_fronier", 0.9)
            log.info(scheduler.get_hypervolume())
            # Frontier dataframe
            params_df = pd.DataFrame(
                sobol_frontier.param_dicts,
                index=range(cfg.meta.n_pareto_points)
            )
            metrics_df = pd.DataFrame(
                [{k:v[i] for k,v in sobol_frontier.means.items()} for i in range(cfg.meta.n_pareto_points)],
                index=range(cfg.meta.n_pareto_points)
            )
            sems_df = pd.DataFrame(
                [{k+"_sems":v[i] for k,v in sobol_frontier.sems.items()} for i in range(cfg.meta.n_pareto_points)],
                index=range(cfg.meta.n_pareto_points)
            )
            df = pd.merge(params_df, metrics_df, left_index=True, right_index=True)
            df = pd.merge(df, sems_df, left_index=True, right_index=True)
            df.to_csv(f"{cfg.problem.name}_frontier_report.csv")
        except:
            log.warning("Could not plot paleto front, not a multi-objective optimization?")

    # Feature Importance
    try:
        cur_model = scheduler.generation_strategy.model
        feature_importance = plot_feature_importance_by_feature(cur_model, relative=True)
        render(feature_importance)
        with open(f'{cfg.problem.name}_feature_importance.html', 'w') as outfile:
            outfile.write(render_report_elements(
                f"{cfg.problem.name}_feature_importance", 
                html_elements=[plot_config_to_html(feature_importance)], 
                header=False,
            ))
        importances = pd.DataFrame([{j['y'][k]:j['x'][k] for k in range(len(cfg.problem.parameters.keys()))} for j in feature_importance.data['data']],
            index=[feature_importance.data['layout']['updatemenus'][0]['buttons'][i]['label'] for i in range(len(feature_importance.data['data']))])
        importances.to_csv(f"{cfg.problem.name}_feature_importance_report.csv")
        log.info(importances)
    except:
        log.warning("Could not compute feature importance, no Gaussian process has been called?")

if __name__ == "__main__":
    exp_main()
