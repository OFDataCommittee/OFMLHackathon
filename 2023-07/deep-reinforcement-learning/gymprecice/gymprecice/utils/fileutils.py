"""A set of common utilities used for file management within the environments.

These are not intended as API functions, and will not remain stable over time.
"""
import json
import logging
import os
import shutil
from datetime import datetime
from os.path import join
from time import sleep
from typing import List, Optional, TextIO

from gymprecice.utils.constants import MAX_ACCESS_WAIT_TIME, SLEEP_TIME
from gymprecice.utils.xmlutils import replace_keyword


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def make_env_dir(env_dir: str = None, solver_list: List = None) -> None:
    """Create a directory with all necessary solver and config files to represent a full training environment.

    Args:
        env_dir (str): envirenment directory path
        solver_list (list): list of solvers reside in the environment.
    """
    os.system(f"rm -rf {os.path.join(os.getcwd(), env_dir)}")
    for solver in solver_list:
        solver_case_dir = os.path.join(os.getcwd(), solver)
        try:
            if os.path.isdir(solver_case_dir):
                os.makedirs(os.path.join(os.getcwd(), env_dir, solver))
                os.system(f"cp -rs {solver_case_dir} {env_dir}")
            else:
                raise OSError
        except Exception as err:
            logger.error("Failed to create symbolic links to solver files")
            raise err
    sleep(SLEEP_TIME)


def open_file(file: str = None) -> TextIO:
    """Open dynamic files."""
    max_attempts = int(MAX_ACCESS_WAIT_TIME / 1e-6)
    acceess_counter = 0
    while True:
        try:
            file_object = open(file)
            break
        except OSError:
            acceess_counter += 1
            if acceess_counter < max_attempts:
                continue
            else:
                # break after trying max_attempts
                raise OSError(f"Could not access {file} after {max_attempts} attempts")
    return file_object


def make_result_dir(
    time_stamped: Optional[bool] = True, suffix: Optional[str] = None
) -> dict:
    r"""Create a directory to save train/prediction results.

    Args:
        time_stamped (optional bool): If the directory name gets time-stamped.
        Warning regarding Data-loss: When set to False, the function will overwrite any directory with the same environment name in `gymprecice-run`,
        unless a distinctive suffix is provided.
        suffix (optional str): if time_stamped is False, then add the suffix to result directory name.


    Note:
        "precice-config.xml" is the precice configuration file that should be located in "physics-simulation-engine" directory of your problem case.\n
        "gymprecice-config.json" is the environment configuration file that should be located in "physics-simulation-engine" directory of your problem case.

        "gymprecice-config.json" has the following format: \n

        {
            "environment": {
                "name": "",
                "result_save_path": "",  // This keyword is optional
            },
            "solvers": {
                "name": [],
                "reset_script": "",
                "run_script": "",
            },
            "actuators": {
                "name": []
            }
        }
    """
    run_dir = None
    precice_config_name = "precice-config.xml"
    gymprecice_config_name = "gymprecice-config.json"

    sim_engine = join(os.getcwd(), "physics-simulation-engine")
    precice_config = join(sim_engine, precice_config_name)
    gymprecice_config = join(sim_engine, gymprecice_config_name)

    with open(gymprecice_config) as config_file:
        content = config_file.read()
    options = json.loads(content)
    options.update({"precice": {"config_file": precice_config_name}})

    result_path = options["environment"].get("results_path", os.getcwd())
    env_name = options["environment"]["name"]
    solver_names = options["physics_simulation_engine"]["solvers"]
    solver_dirs = [join(sim_engine, solver) for solver in solver_names]

    if time_stamped:
        time_str = datetime.now().strftime("%d%m%Y_%H%M%S")
        run_dir = join(result_path, "gymprecice-run", f"{env_name}_{time_str}")
    else:
        if suffix is None:
            run_dir = join(result_path, "gymprecice-run", f"{env_name}")
        else:
            run_dir = join(result_path, "gymprecice-run", f"{env_name}_{suffix}")

    try:
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir, exist_ok=True)
    except Exception as err:
        logger.error(f"Failed to create {run_dir}")
        raise err

    try:
        for solver_dir in solver_dirs:
            os.system(f"cp -r {solver_dir} {run_dir}")
    except Exception as err:
        logger.error(f"Failed to copy base case to {run_dir}")
        raise err

    try:
        os.system(f"cp {precice_config} {run_dir}")
    except Exception as err:
        logger.error(f"Failed to copy precice config file to {run_dir}")
        raise err

    os.chdir(str(run_dir))

    keyword = "exchange-directory"
    keyword_value = f"{run_dir}/precice-{keyword}"
    replace_keyword(
        precice_config_name,
        keyword,
        keyword_value,
        place_counter_postfix=True,
    )

    return options
