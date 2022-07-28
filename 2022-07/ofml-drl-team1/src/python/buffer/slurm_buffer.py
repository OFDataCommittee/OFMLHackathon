
from src.python.buffer.local_buffer import LocalBuffer
from os.path import join
from shutil import copytree
from copy import deepcopy
import torch as pt
from .buffer import Buffer
from subprocess import Popen, PIPE
from queue import Queue
from _thread import start_new_thread
import random
import os
from time import sleep

class SlurmBuffer(LocalBuffer):
    def __init__(self,  path: str, env, buffer_size: int, n_runners: int, n_cores: int):
        LocalBuffer.__init__(self, path, env, buffer_size, n_runners)
        self._n_cores = n_cores

    def write_jobfile(self, core_count, job_name, job_dir):
        with open(f'{job_dir}/jobscript.sh', 'w') as rsh:
            rsh.write(f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --ntasks={core_count}
#SBATCH --output={job_name}.out
#SBATCH --partition=c6i
#SBATCH --constraint=c6i.32xlarge
source activate pydrl
module load openmpi
source /fsx/OpenFOAM/OpenFOAM-v2206/etc/bashrc
cd {job_dir}
./Allrun""")

        os.system(f"chmod +x {job_dir}/jobscript.sh")


    def job_wait(self, job_id, proc, job_name, queue):
        running = True
        while running:
            try:
                p = Popen(['squeue', '-j', f"{job_id}"], stdout=PIPE)
                jobstatus = str(p.stdout.read(), 'utf-8').split()[12]
                if jobstatus=='PD' or jobstatus=='R' or jobstatus=='CF':
                    sleep(5)
                else:
                    queue.put((job_name, '0'))
                    running = False
            except Exception as e:
                queue.put((job_name, '0'))
                running = False

    def fill_slurm(self):
        count_running, count_started = 0, 0
        queue = Queue()
        proc = []

        # start initial batch of simulations
        for i in range(min(self._n_runners, self._buffer_size)):

            self.write_jobfile(self._n_cores, job_name=f'traj_{i}', job_dir=f"{self._path}/runner_{count_started}")
            proc.append(Popen(['sbatch', 'jobscript.sh'], cwd=f"{self._path}/runner_{count_started}", stdout=PIPE))
            output = str(proc[-1].stdout.read(), 'utf-8')
            output = output.replace('Submitted batch job ', '')
            output = output.replace('/n', '')
            jobid = int(output)

            start_new_thread(self.job_wait, (jobid, proc[-1], f"job_{count_started}", queue))
            count_running += 1
            count_started += 1

        while count_running > 0:
            # wait for completion
            job_name = queue.get()
            #print(f"{job_name} finished ")

            # start more simulations if necessary
            if self._buffer_size > count_started:
                self.write_jobfile(self._n_cores, job_name=f'traj_{i}', job_dir=f"{self._path}/runner_{count_started}")
                proc.append(Popen(['sbatch', 'jobscript.sh'], cwd=f"{self._path}/runner_{count_started}", stdout=PIPE))
                output = str(proc[-1].stdout.read(), 'utf-8')
                output = output.replace('Submitted batch job ', '')
                output = output.replace('/n', '')
                jobid = int(output)
                start_new_thread(self.job_wait, (jobid, proc[-1], f"job_{count_started}", queue))

                count_started += 1
                count_running += 1
            count_running -= 1

    def sample(self):
        for env in self._envs:
            s, a, r, l = env.observations
            self._states.append(s[1:])
            self._actions.append(a[1:-1])
            self._rewards.append(r[:])
            self._log_p.append(l[1:-1])
        return self._states, self._actions, self._rewards, self._log_p
