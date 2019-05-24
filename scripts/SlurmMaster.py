#!/usr/bin/env python

# Inspired by Nate Odell's (naodell@gmail.com) BatchMaster.py for condor
# https://github.com/NWUHEP/BLT/blob/topic_wbranch/BLTAnalysis/python/BatchMaster.py

import sys
import os
from pathlib import Path
import configparser
import json
import subprocess
from datetime import datetime


class SlurmMaster:
    def __init__(self):
        self.date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = Path('/pylon5/ac5616p/bpollack/mre_slurm', self.date)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def generate_slurm_script(self, number):
        '''Make a slurm submission script.'''
        script = open(f'./slurm_script_{self.date}_n{number}', 'w')
        script.write('#!/bin/bash\n')
        script.write('#SBATCH -A ac5616p\n')
        script.write('#SBATCH --partition=GPU-AI\n')
        script.write('#SBATCH --nodes=1\n')
        script.write('#SBATCH --gres=gpu:volta16:1\n')
        script.write('#SBATCH --time=1:00:00\n')
        script.write('#SBATCH --mail-user=brianleepollack@gmail.com\n')
        script.write(f'#SBATCH --output={str(self.log_dir)}/job_n{number}.stdout\n')
        script.write(f'#SBATCH --error={str(self.log_dir)}/job_n{number}.stderr\n')
        script.write('\n')

        script.write('set -x\n')
        script.write('echo "$@"\n')
        script.write('source /pghbio/dbmi/batmanlab/bpollack/anaconda3/etc/profile.d/conda.sh\n')
        script.write('conda activate new_mre\n')

        script.close()
        return script

    def parse_config(self):
        config = configparser.ConfigParser()
        config.read('test_config.ini')
        section = config.sections()[0]
        self.config_dict = {}
        for c in config[section]:
            print(c, json.loads(config[section][c]))


if __name__ == "__main__":
    SM = SlurmMaster()
    # SM.generate_slurm_script(0)
    SM.parse_config()
