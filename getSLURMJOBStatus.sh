#!/usr/bin/bash
# ! WARNING ! this might not be as reliable as sacct
scontrol show job | awk '/JobName='$1'/{a=1} (a==1 && /JobState=/){gsub("=", " ", $1); print($1)}' | tac
