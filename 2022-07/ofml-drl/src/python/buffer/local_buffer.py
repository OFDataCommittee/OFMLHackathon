
from os.path import join
import os
from shutil import copytree
from copy import deepcopy
import subprocess
import _thread
import queue
import torch as pt
from .buffer import Buffer
import numpy as np


class LocalBuffer(Buffer):
    def __init__(self, path: str, env, size: int, n_runners: int):
        self._path = path
        self._base_env = env
        self._size = size
        self._n_runners = n_runners

        self._envs = self._create_copies()
        self._states, self._actions, self._rewards, self._log_p = [], [], [], []

    def _create_copies(self):
        envs = []
        for i in range(self._size):
            dest = join(self._path, f"runner_{i}")
            copytree(self._base_env.path, dest, dirs_exist_ok=True)
            envs.append(deepcopy(self._base_env))
            envs[-1].path = dest
        return envs

    def fill(self):
        # TODO: If fewer runners than buffer size, this needs reimplementation to be different for each buffer entry
        for i, env in enumerate(self._envs):
            env.seed = i
        
        # run case
        # get status of trajectory
        results = queue.Queue()
        process_count = 0
        proc = []

        # set the n_workers
        print("Buffer size: ", self._size)
        for t in range(int(max(self._size, self._n_runners))):
            item = "proc_" + str(t)
            proc.append(item)
        print(proc)

        # execute the n = n_workers trajectory simultaneously
        # set the counter to count the number of trajectory
        buffer_counter = 0
        for n in np.arange(self._n_runners):
            #self.run_trajectory(buffer_counter, proc, results, sample, action_bounds, env)
            self.run_trajectory2(buffer_counter, proc, results, self._envs[buffer_counter])
            process_count += 1
            # increase the counter of trajectory number
            buffer_counter += 1

        # fetch observations
        # check for any worker is done. if so give next trajectory to that worker
        while process_count > 0:
            job_name, rc = results.get()
            print("job : ", job_name, "finished with rc =", rc)
            if self._size > buffer_counter:
                #self.run_trajectory(buffer_counter, proc, results, sample, action_bounds, env)
                self.run_trajectory2(buffer_counter, proc, results, self._envs[buffer_counter])
                process_count += 1
                buffer_counter += 1
            process_count -= 1

    def sample(self):
        return self._states, self._actions, self._rewards, self._log_p

    def update_policy(self, policy):
        for env in self._envs:
            policy.save(join(env.path, env.policy))

    def reset(self):
        for env in self._envs:
            env.reset()
        self._states, self._actions, self._rewards, self._log_p = [], [], [], []

    # Added helper code
    def process_waiter(self, proc, job_name, que):
        """
             This method is to wait for the executed process till it is completed
         """
        try:
            proc.wait()
        finally:
            que.put((job_name, proc.returncode))

    def run_trajectory(self, buffer_counter, proc, results, sample, action_bounds, env):
        """
        To run the trajectories
        Args:
            buffer_counter: which trajectory to run (n -> traj_0, traj_1, ... traj_n)
            proc: array to hold process waiting flag
            results: array to hold process finish flag
            sample: number of iteration of main ppo
            action_bounds: min and max omega value
        Returns: execution of OpenFOAM Allrun file in machine
        """
        # # some hardcoded trajectory settings
        # core_count = 2

        # # make dir for new trajectory
        # traj_path = f"./env/sample_{sample}/trajectory_{buffer_counter}"

        # print(f"\n starting trajectory : {buffer_counter} \n")
        # os.makedirs(traj_path, exist_ok=True)
        # # copy files form base_case
        # # change of ending time -> system/controlDict
        # os.popen(
        #     f'cp -r ./env/base_case/agentRotatingWallVelocity/* {traj_path}/ &&'
        #     f'sed -i "s/timeStart.*/timeStart       4.01;/g" {traj_path}/system/controlDict &&'
        #     f'sed -i "/^endTime/ s/endTime.*/endTime         5.0;/g" {traj_path}/system/controlDict'
        # )
        
        # for i in range(core_count):
        #     os.popen(
        #         f'sed -i "s/startTime.*/startTime       4.009999;/g" {traj_path}/processor{i}/4/U &&'
        #         f'sed -i "s/absOmegaMax.*/absOmegaMax       {action_bounds[1]};/g" {traj_path}/processor{i}/4/U'
        #     )
        # executing Allrun to start trajectory
        proc[buffer_counter] = subprocess.Popen([f'{env.run_script}'], cwd=f'{env.path}/')
        _thread.start_new_thread(self.process_waiter,
                                 (proc[buffer_counter], f"trajectory_{buffer_counter}", results))

    def run_trajectory2(self, buffer_counter, proc, results, env):
        """
        To run the trajectories
        Args:
            buffer_counter: which trajectory to run (n -> traj_0, traj_1, ... traj_n)
            proc: array to hold process waiting flag
            results: array to hold process finish flag
            sample: number of iteration of main ppo
            action_bounds: min and max omega value
        Returns: execution of OpenFOAM Allrun file in machine
        """
        # # some hardcoded trajectory settings
        # core_count = 2

        # # make dir for new trajectory
        # traj_path = f"./env/sample_{sample}/trajectory_{buffer_counter}"

        # print(f"\n starting trajectory : {buffer_counter} \n")
        # os.makedirs(traj_path, exist_ok=True)
        # # copy files form base_case
        # # change of ending time -> system/controlDict
        # os.popen(
        #     f'cp -r ./env/base_case/agentRotatingWallVelocity/* {traj_path}/ &&'
        #     f'sed -i "s/timeStart.*/timeStart       4.01;/g" {traj_path}/system/controlDict &&'
        #     f'sed -i "/^endTime/ s/endTime.*/endTime         5.0;/g" {traj_path}/system/controlDict'
        # )
        
        # for i in range(core_count):
        #     os.popen(
        #         f'sed -i "s/startTime.*/startTime       4.009999;/g" {traj_path}/processor{i}/4/U &&'
        #         f'sed -i "s/absOmegaMax.*/absOmegaMax       {action_bounds[1]};/g" {traj_path}/processor{i}/4/U'
        #     )
        print(f"CWD: {env.path}")
        print(f"CWD absolute: {join(os.getcwd(), env.path)}")
        print(f"Buffer counter: {buffer_counter}")
        # executing Allrun to start trajectory
        proc[buffer_counter] = subprocess.Popen(f'./wait.sh', cwd=f'{join(os.getcwd(), env.path)}')
        #proc[buffer_counter] = subprocess.Popen([f'{env.run_script}'], cwd=f'{join(os.getcwd(), env.path)}')
       
        _thread.start_new_thread(self.process_waiter,
                                 (proc[buffer_counter], f"runner_{buffer_counter}", results))