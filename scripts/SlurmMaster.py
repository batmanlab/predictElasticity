#!/usr/bin/env python

# Inspired by Nate Odell's (naodell@gmail.com) BatchMaster.py for condor
# https://github.com/NWUHEP/BLT/blob/topic_wbranch/BLTAnalysis/python/BatchMaster.py

import sys
import os
import shutil
import random
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
        self.notes_dir = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/notes',
                              self.date)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.config = Path(config)
        self.parse_config()
        random.seed(self.date)
        self.mre_id = random.randint(10000, 90000)
        print('making staging dir')
        os.mkdir(f'/pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/{self.date}')
        shutil.copytree('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/mre',
                        '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/' +
                        f'{self.date}/mre{self.mre_id}')
        shutil.copy('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/__init__.py',
                    '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/' +
                    f'{self.date}/__init__.py')
        shutil.copy('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/setup.py',
                    '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/' +
                    f'{self.date}/setup.py')
        shutil.rmtree('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/' +
                      f'{self.date}/mre{self.mre_id}/__pycache__')
        os.system(f"sed -i 's/import mre/import mre{self.mre_id}/g' " +
                  "/pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/" +
                  f'{self.date}/mre{self.mre_id}/*.py')
        os.system(f"sed -i 's/from mre/from mre{self.mre_id}/g' " +
                  "/pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/" +
                  f"{self.date}/mre{self.mre_id}/*.py")
        os.system(f"sed -i 's/mre/mre{self.mre_id}/g' " +
                  "/pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/" +
                  f"{self.date}/setup.py")
        os.chdir(f'/pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/{self.date}')
        os.system('python setup.py develop')

    def generate_slurm_script(self, number, conf, subj, subj_num, date, project):
        '''Make a slurm submission script.'''
        if project == 'MRE':
            module = 'train_mre_model.py'
            self.gpu = True
        elif project == 'CHAOS':
            module = 'train_seg_model.py'
            self.gpu = True
        elif project == 'XR':
            module = 'make_xr.py'
            self.gpu = False

        if type(subj) is list:
            subj_name = f'GROUP{subj_num}'
            subj = ' '.join(subj)
        else:
            subj_name = subj

        print(conf)
        arg_string = ''
        for i in conf:
            if type(conf[i]) is list:
                clean_vals = ' '.join(conf[i])
                arg_string += f' --{i} {clean_vals}'
            else:
                arg_string += f' --{i}={conf[i]}'
        # arg_string = ' '.join(f'--{i}={conf[i]}' for i in conf)
        script_name = f'/tmp/slurm_script_{self.date}_n{number}_subj{subj_name}'
        script = open(script_name, 'w')
        script.write('#!/bin/bash\n')
        if self.gpu:
            arg_string += f' --subj {subj} --subj_group={subj_name}'
            arg_string += f' --model_version={date}_n{number}'
            # script.write('#SBATCH -D /pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging')
            script.write('#SBATCH -A ac5616p\n')
            script.write(f'#SBATCH --partition={self.node["partition"]}\n')
            # script.write('#SBATCH --gres=gpu:volta32:1\n')
            script.write(f'#SBATCH --gres=gpu:{self.node["gpu"]}:{self.node["ngpus"]}\n')
            # script.write('#SBATCH -A bi561ip\n')
            # script.write('#SBATCH --partition=DBMI-GPU\n')
            # script.write('#SBATCH --gres=gpu:p100:2\n')
            script.write('#SBATCH --nodes=1\n')
            # script.write('#SBATCH --mem=92800\n')
            script.write('#SBATCH -C EGRESS\n')
        else:
            arg_string += f' --subj={subj}'
            script.write('#SBATCH -A bi561ip\n')
            script.write('#SBATCH --partition=DBMI\n')
            script.write('#SBATCH --mem=120GB\n')
            script.write('#SBATCH -C EGRESS\n')
        script.write('#SBATCH --time=24:00:00\n')
        script.write('#SBATCH --mail-user=brianleepollack@gmail.com\n')
        script.write(f'#SBATCH --output={str(self.log_dir)}/job_n{number}_subj{subj_name}.stdout\n')
        script.write(f'#SBATCH --error={str(self.log_dir)}/job_n{number}_subj{subj_name}.stderr\n')
        script.write('\n')

        script.write('set -x\n')
        script.write('echo "$@"\n')
        script.write('source /pghbio/dbmi/batmanlab/bpollack/anaconda3/etc/profile.d/conda.sh\n')
        script.write('conda activate mre\n')
        # script.write('python /pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/' +
        #              f'{self.date}/setup.py install\n')
        script.write('\n')
        script.write('nvidia-smi\n')

        script.write(f'python /pghbio/dbmi/batmanlab/bpollack/predictElasticity/staging/'
                     f'{self.date}/mre{self.mre_id}/{module} {arg_string}\n')

        script.close()
        print(arg_string)
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

        if 'Notes' in sections:
            self.notes = config['Notes']['note']
            with open(Path(self.notes_dir, 'notes.txt'), 'w') as f:
                f.writelines(self.notes+'\n')

        if 'Node' in sections:
            self.node = config['Node']

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
            for j, subj in enumerate(self.subj_list):
                script_name = self.generate_slurm_script(0, self.config_dict, subj, j, self.date,
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
    print(SM.notes)
