#!/usr/bin/env python

# Inspired by Nate Odell's (naodell@gmail.com) BatchMaster.py for condor
# https://github.com/NWUHEP/BLT/blob/topic_wbranch/BLTAnalysis/python/BatchMaster.py

import sys
import os
from pathlib import Path
import argparse
import configparser
import json
import ast
import subprocess
import itertools
from datetime import datetime


class SlurmMaster:
    def __init__(self, config):
        self.date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = Path('/pylon5/ac5616p/bpollack/mre_slurm', self.date)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = Path(config)
        self.parse_config()

    def generate_slurm_script(self, number, conf, subj, subj_num, date, project):
        '''Make a slurm submission script.'''
        if project == 'MRE':
            module = 'train_model_full_v2.py'
            self.gpu = True
        elif project == 'CHAOS':
            module = 'train_seg_model.py'
            self.gpu = True
        elif project == 'XR':
            module = 'make_xr.py'
            self.gpu = False

        if type(subj) is list:
            subj_name = f'GROUP{subj_num}'
        else:
            subj_name = subj

        arg_string = ' '.join(f'--{i}={conf[i]}' for i in conf)
        script_name = f'/tmp/slurm_script_{self.date}_n{number}_subj{subj_name}'
        script = open(script_name, 'w')
        script.write('#!/bin/bash\n')
        if self.gpu:
            arg_string += f' --subj={subj} --model_version={date}_n{number}'
            script.write('#SBATCH -A ac5616p\n')
            script.write('#SBATCH --partition=GPU-AI\n')
            script.write('#SBATCH --nodes=1\n')
            script.write('#SBATCH -C EGRESS\n')
            script.write('#SBATCH --gres=gpu:volta16:4\n')
        else:
            arg_string += f' --subj={subj}'
            script.write('#SBATCH -A bi561ip\n')
            script.write('#SBATCH --partition=DBMI\n')
            script.write('#SBATCH --mem=120GB\n')
            script.write('#SBATCH -C EGRESS\n')
        script.write('#SBATCH --time=8:00:00\n')
        script.write('#SBATCH --mail-user=brianleepollack@gmail.com\n')
        script.write(f'#SBATCH --output={str(self.log_dir)}/job_n{number}_subj{subj_name}.stdout\n')
        script.write(f'#SBATCH --error={str(self.log_dir)}/job_n{number}_subj{subj_name}.stderr\n')
        script.write('\n')

        script.write('set -x\n')
        script.write('echo "$@"\n')
        script.write('source /pghbio/dbmi/batmanlab/bpollack/anaconda3/etc/profile.d/conda.sh\n')
        script.write('conda activate new_mre\n')
        script.write('\n')

        script.write(f'python /pghbio/dbmi/batmanlab/bpollack/predictElasticity/'
                     f'mre/{module} {arg_string}\n')

        script.close()
        return script_name

    def parse_config(self):
        config = configparser.ConfigParser()
        # config.read('config_inis/test_config.ini')
        config.read(str(self.config))
        sections = config.sections()
        self.config_dict = {}
        self.subj_list = []
        self.project = None

        if 'Project' in sections:
            self.project = config['Project']['task']
        else:
            self.project = 'MRE'

        # Iterate through config and convert all scalars to lists
        for c in config['Hyper']:
            print(c)
            print(config['Hyper'][c])
            val = ast.literal_eval(config['Hyper'][c])
            if c == 'subj':
                if type(val) == list:
                    self.subj_list  = val
                else:
                    self.subj_list.append(val)
            elif c == 'subj_group':
                self.subj_list = val
            else:
                if type(val) == list:
                    self.config_dict[c] = val
                else:
                    self.config_dict[c] = [val]
        if len(self.subj_list) == 0:
            self.subj_list.append('162')

        # Make every possible combo of config items
        if self.project != 'XR':
            self.config_combos = product_dict(**self.config_dict)

    def submit_scripts(self):
        if self.project != 'XR':
            for i, conf in enumerate(self.config_combos):
                for j, subj in enumerate(self.subj_list):
                    script_name = self.generate_slurm_script(i, conf, subj, j, self.date,
                                                             self.project)
                    print(script_name)
                    subprocess.call(f'sbatch {script_name}', shell=True)
        else:
            for subj in self.subj_list:
                script_name = self.generate_slurm_script(0, self.config_dict, subj, self.date,
                                                         self.project)
                print(script_name)
                subprocess.call(f'sbatch {script_name}', shell=True)


def product_dict(**kwargs):
    '''From https://stackoverflow.com/a/5228294/4942417,
    Produce all combos of configs for list-like items.'''
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Submit a series of SLURM jobs.')
    parser.add_argument('config', type=str, help='Path to config_file.')
    args = parser.parse_args()

    SM = SlurmMaster(args.config)
    SM.submit_scripts()
