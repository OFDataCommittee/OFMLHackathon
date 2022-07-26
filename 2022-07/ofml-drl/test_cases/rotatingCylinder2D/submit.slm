#!/bin/bash -l

sentence=$(sbatch $1)              # get the output from sbatch

stringarray=($sentence)                            # separate the output in words
jobid=(${stringarray[3]})                          # isolate the job ID
sentence="$(squeue -j $jobid)"       		   # read job's slurm status
stringarray=($sentence)
jobstatus=(${stringarray[12]})            	   # isolate the status of job number jobid
message=(${stringarray[10]})

while [ "$jobstatus" = "R" ] || [ "$jobstatus" = "PD" ];
do
  echo "waiting for $message ..."
  sentence="$(squeue -j $jobid)"                     # read job's slurm status
  stringarray=($sentence)
  jobstatus=(${stringarray[12]})                     # isolate the status of job number jobid
  sleep 03
done
