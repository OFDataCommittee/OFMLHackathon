#!/usr/bin/python3
"""Core functionality for performing paramter variation and optimization on OpenFOAM cases"""

import os, hashlib, shutil, logging
import subprocess as sb
import regex as re
from collections import defaultdict

import pandas as pd
import numpy as np

from omegaconf import OmegaConf, DictConfig, DictKeyType
from ax.core.base_trial import TrialStatus, BaseTrial
from ax.core.trial import Trial
from ax.core.runner import Runner
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.data import Data
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.notebook.plotting import plot_config_to_html, render
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.utils.report.render import render_report_elements
from typing import Any, Dict, NamedTuple, Union, Iterable, Set, List
from ax.utils.common.result import Ok, Err
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.storage.json_store.registry import CORE_ENCODER_REGISTRY, CORE_DECODER_REGISTRY, CORE_CLASS_DECODER_REGISTRY
from ax.storage.json_store.encoders import metric_to_dict
from ax.storage.json_store.encoders import runner_to_dict

from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

log = logging.getLogger(__name__)

def process_input_command(command, case):
    """
        Process commands from config files to provide some flexibility
    """
    return [c.replace("$CASE_PATH", case.name).replace("$CASE_NAME", os.path.basename(case.name)) for c in command]

def gen_search_space(cfg):
    """
        Generate a search space for Ax from Hydra config object.
        Looks for "parameters" entry in passed-in config object.
    """

    l = []
    for key, item in cfg.parameters.items():
        e = {
            "name": key,
            **item
        }
        if 'values' in e.keys():
            e['values'] = list(e['values'])
        if 'bounds' in e.keys():
            e['bounds'] = list(e['bounds'])
        if 'dependents' in e.keys():
            #e['dependents'] = list(e['dependents'])
            print("#######", e)
            tmp = {}
            for k in e['dependents']:
                for kk in k.keys():
                    tmp.update({kk: list(k[kk])})
            e['dependents'] = tmp
            print("########", e)
        l.append(e)    
    return l

def gen_objectives(cfg):
    """
        Generate objectives for Multi-objective optimization studies from Hydra config object
        Looks for "objectives" entry in passed-in config object.
    """

    objs = {}
    for key, item in cfg.objectives.items():
        objs[key] = ObjectiveProperties(minimize=item.minimize, threshold=item.threshold)
    return objs

def plot_frontier(frontier,name, CI_level=0.9):
    """
        Plot pareto frontier with CI_level error bars into an HTML file.
    """

    plot_config = plot_pareto_frontier(frontier, CI_level=CI_level)
    with open(f'{name}_report.html', 'w') as outfile:
        outfile.write(render_report_elements(
            f"{name}_report", 
            html_elements=[plot_config_to_html(plot_config)], 
            header=False,
            ))
    render(plot_config)

class HPCJob(NamedTuple):
    """
        An async OpenFOAM job scheduled on an HPC system.
    """

    id: int
    parameters: Dict[str, Union[str, float, int, bool]]
    mode: str
    config: Dict[str, Any]

def local_case_run(parameters, case, cfg):
    """
        Run shell command on local machine; but does not wait for completion.
    """
    proc = sb.Popen(list(process_input_command(cfg.meta.case_run_command, case)), cwd=case.name, stdout=sb.PIPE, stderr=sb.PIPE)
    job_id = proc.pid
    job = HPCJob(id=job_id, parameters=parameters, mode=cfg.meta.case_run_mode, config={"local": proc, "case": case})
    return (job_id, job)

def slurm_case_run(parameters, case, cfg):
    """
        Run SLURM batch job
    """
    curr_cwd = os.getcwd()
    os.chdir(case.name)
    proc_out = sb.check_output(list(process_input_command(cfg.meta.case_run_command, case)), cwd=case.name, stderr=sb.PIPE)
    os.chdir(curr_cwd)
    job_id = -1
    match = re.search(b"[0-9]+", proc_out)
    if match:
        sub_string = match.group()
        job_id = int(sub_string)
    else:
        log.warn(f"SLURM command: '{cfg.meta.case_run_command}' did not submit a job?")
    job = HPCJob(id=job_id, parameters=parameters, mode=cfg.meta.case_run_mode, config={"slurm": proc_out, "case": case})
    return (job_id, job)

def local_status_query(job_id, jobs, cfg):
    """
        Check for job status locally, by polling the Popen object
    """
    job = jobs[job_id].config["local"]
    job.poll()
    if  job.returncode is None:
        return TrialStatus.RUNNING
    elif job.returncode == 0:
        return TrialStatus.COMPLETED
    else:
        return TrialStatus.FAILED

def slurm_status_query(job_id, jobs, cfg):
    """
        Check for job status on the slurm cluster, via a user-supplied command

        Expected output:
            job_id STATUS
        or
            job_name STATUS
        if multiple raws are supplied (eg. step status), only the first one is considered
    """
    case = jobs[job_id].config["case"]
    proc_out = sb.check_output(list(process_input_command(cfg.meta.slurm_status_query, case)), cwd=case.name, stderr=sb.PIPE)
    # TODO: I have no idea why this isnecessary, but sacct sometimes returns an empty string!
    # Hash clashes maybe?
    if proc_out.decode("utf-8") == "":
        return TrialStatus.COMPLETED
    status = str(proc_out.split()[1].decode("utf-8"))
    status_map = {
        "RUNNING": TrialStatus.RUNNING,
        "CONFIGURING": TrialStatus.RUNNING,
        "COMPLETING": TrialStatus.RUNNING,
        "PENDING": TrialStatus.RUNNING,
        "PREEMPTED": TrialStatus.FAILED,
        "FAILED": TrialStatus.FAILED,
        "SUSPENDED": TrialStatus.ABANDONED,
        "TIMEOUT": TrialStatus.ABANDONED,
        "STOPPED": TrialStatus.EARLY_STOPPED,
        "CANCELED": TrialStatus.EARLY_STOPPED,
        "CANCELLED+": TrialStatus.EARLY_STOPPED,
        "COMPLETED": TrialStatus.COMPLETED,
    }
    return status_map[status]

def shell_metric_value(metric, case, cfg):
    """
        Run a bash command to extract a single metric/objective value.
        The command runs inside the OpenFOAM case if $CASE_PATH doesn't show up in it

        Optionally, a preparation command can be supplied for HPC runs if necessary.
        Note that the these commands need to be interactive (eg. salloc) as we need to
        block and figure out the objective's value at this point.
    """
    metrics = {}
    item = cfg.problem.objectives[metric]
    # OpenFOAM is annoying in this regard, so, if OpenFOAM utils are used to
    # extract metrics, do a: foamUtility -case $CASEPATH
    hasPath=any([c.find('$CASE_PATH') != -1 for c in item.command])
    cwd = os.getcwd() if hasPath else case.name
    try:
        if "prepare" in item.keys():
            sb.check_output(list(process_input_command(item.prepare, case)), cwd=cwd)
    except:
        log.warning(f"prepare command was not successful for {item}")
    try:
        out = sb.check_output(list(process_input_command(item.command, case)), cwd=cwd)
        metrics[metric] = float(out)
    except:
        log.warning(f"Metric output for {item} cannot be converted to float, considering NaN...")
        metrics[metric] = np.nan
    return metrics

class HPCJobQueueClient:
    """
        A job queue when the `Scheduler` will
        deploy trial evaluation runs during optimization.
    """

    jobs: Dict[int, HPCJob] = {}
    cfg: Any
    dispatcher_map = {
        "local": local_case_run,
        "slurm": slurm_case_run,
    }
    status_query_map = {
        "local": local_status_query,
        "slurm": slurm_status_query,
    }
    metrics_value_map = {
        "shell": shell_metric_value,
    }

    def schedule_job_with_parameters(
            self, parameters: Dict[str, Union[str, float, int, bool]], case, cfg
            ) -> int:
        """
            Schedules an evaluation job with given parameters and returns job ID.
        """
        job_id = None
        self.cfg = cfg
        job_id, job = self.dispatcher_map[cfg.meta.case_run_mode](parameters, case, cfg)
        self.jobs[job_id] = job
        return job_id

    def get_job_status(self, job_id: int) -> TrialStatus:
        """
            Get status of the job by a given ID. Will return an Ax `TrialStatus`
        """
        return self.status_query_map[self.cfg.meta.case_run_mode](job_id, self.jobs, self.cfg)

    def get_outcome_value_for_completed_job(self, job_id: int) -> Dict[str, float]:
        """
            Run metric evaluation commands on finished jobs.
        """
        metrics = {}
        case = self.jobs[job_id].config["case"]
        # Dispatch a way to get metric; can be different for each metric
        for key, item in self.cfg.problem.objectives.items():
            metrics.update(self.metrics_value_map[self.cfg.problem.objectives[key].mode](key, case, self.cfg))
        return metrics

HPC_JOB_QUEUE_CLIENT = HPCJobQueueClient()

def get_hpc_job_queue_client() -> HPCJobQueueClient:
    """Obtain the singleton job queue instance."""
    return HPC_JOB_QUEUE_CLIENT


def preprocesss_case(parameters, cfg):
    """
        Copy template, and substitute parameter values
    """
    data = {}

    # Hash parameters to avoid long trial names
    hash = hashlib.md5()
    encoded = repr(OmegaConf.to_yaml(parameters)).encode()
    hash.update(encoded)

    # Clone template case
    templateCase = SolutionDirectory(f"{cfg.problem.template_case}", archive=None, paraviewLink=False)
    for d in cfg.meta.case_subdirs_to_clone:
        templateCase.addToClone(d)
    newcase = f"{cfg.problem.name}_trial_"+hash.hexdigest()
    data["casename"] = newcase

    # Run the case through the provided command in the config
    case = templateCase.cloneCase(cfg.meta.clone_destination+newcase)
    # Process parameters which require file copying (you can have one parameter per case file)
    if "file_copies" in cfg.meta.keys():
        for elm,elmv in cfg.meta.file_copies.items():
            shutil.copyfile(
                case.name+elmv.template+"."+parameters[elm],
                case.name+elmv.template
            )
    # Process parameters with PyFoam
    for elm,elmv in cfg.meta.scopes.items():
        paramFile = ParsedParameterFile(name=case.name + elm)
        for param in elmv:
            splits = elmv[param].split('.')
            lvl = paramFile[splits[0]]
            if len(splits) > 1:
                for i in range(1,len(splits)-1):
                    scp = splits[i]
                    try:
                        scp = int(splits[i])
                    except:
                        pass
                    lvl = lvl[scp]
                try:
                    lvl[splits[-1]] = parameters[param]
                except:
                    log.warn(f"Couldn't substitute {param} value in its scope, if it's a dependent parameter, you can ignore this warning")
            else:
                try:
                    paramFile[elmv[param]] = parameters[param]
                except:
                    log.warn(f"Couldn't substitute {param} value in its scope, if it's a dependent parameter, you can ignore this warning")
        paramFile.writeFile()
    data["case"] = case
    return data

class HPCJobRunner(Runner):
    """
        A job representation which is ran on HPC
    """
    
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        """
            Deploys a trial on a HPC system.
        """
        if not isinstance(trial, Trial):
            raise ValueError("This runner only handles `Trial`.")

        # Preprocessing using pyFOAM or copying
        case_data = preprocesss_case(trial.arm.parameters, self.cfg)

        hpc_job_queue = get_hpc_job_queue_client()
        trial._properties['casename'] = case_data["casename"]
        job_id = hpc_job_queue.schedule_job_with_parameters(
            parameters=trial.arm.parameters,
            case=case_data["case"],
            cfg=self.cfg
        )
        # This run metadata will be attached to trial as `trial.run_metadata`
        # by the base `Scheduler`.
        log.info(f"Trial {trial.index} - Dispatched case: {case_data['case'].name}")
        return {"job_id": job_id, "case_path": case_data["case"].name, "case_name": case_data["casename"]}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        """
            Checks the status of any non-terminal trials and returns their
            indices as a mapping from TrialStatus to a list of indices. Required
            for runners used with Ax ``Scheduler``.

            NOTE: Does not need to handle waiting between polling calls while trials
            are running; this function should just perform a single poll.
            NOTE: This does not need to include trials that at the time of polling already 
            have a terminal (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        status_dict = defaultdict(set)
        hpc_job_queue = get_hpc_job_queue_client()
        for trial in trials:
            status = hpc_job_queue.get_job_status(
                job_id=trial.run_metadata.get("job_id")
            )
            status_dict[status].add(trial.index)

        return status_dict

class HPCJobMetric(Metric):  # Pulls data for trial from external system.
    """
        Metric observation on completion of scheduled jobs with HPC software
    """

    def __init__(self, name, cfg) -> None:
        super().__init__(name=name)
        self.cfg = cfg
        self.lower_is_better = cfg['problem']['objectives'][name]['lower_is_better']
    
    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
        """
            Obtains data for this metric via fetching it from the queue for a given trial
        """
        if not isinstance(trial, Trial):
            raise ValueError("This metric only handles `Trial`.")

        try: 
            hpc_job_queue = get_hpc_job_queue_client()
            
            metric_data = hpc_job_queue.get_outcome_value_for_completed_job(
                job_id=trial.run_metadata.get("job_id")
            )
            df_dict = [
                {
                    "trial_index": trial.index,
                    "metric_name": self.name,
                    "arm_name": trial.arm.name,
                    "mean": metric_data.get(self.name),
                    "sem": None,
                }
            ]
            return Ok(value=Data(df=pd.DataFrame.from_records(df_dict)))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )

# Serialization/Deserialization of HPCJob Metrics/Runners to andfrom JSON-like objects

def config_to_dict(config: DictConfig) -> Union[Dict[DictKeyType, Any], List[Any], None, str, Any]:
    return OmegaConf.to_object(config)

def config_from_json(config: Dict[str, Any]) -> DictConfig:
    return DictConfig(config)

CORE_CLASS_DECODER_REGISTRY["Type[DictConfig]"] = config_from_json
CORE_ENCODER_REGISTRY[DictConfig] = config_to_dict

CORE_ENCODER_REGISTRY[HPCJobRunner] = runner_to_dict;
CORE_DECODER_REGISTRY["HPCJobRunner"] = HPCJobRunner
register_runner(HPCJobRunner)

CORE_ENCODER_REGISTRY[HPCJobMetric] = metric_to_dict
CORE_DECODER_REGISTRY["HPCJobMetric"] = HPCJobMetric
register_metric(HPCJobMetric)
