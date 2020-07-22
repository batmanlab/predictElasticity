# import os.path as p
# import subprocess
#
# DIR_OF_THIS_SCRIPT = p.abspath(p.dirname(__file__))
# DIR_OF_THIRD_PARTY = p.join(DIR_OF_THIS_SCRIPT, 'third_party')
#
#
# def GetStandardLibraryIndexInSysPath(sys_path):
#     for index, path in enumerate(sys_path):
#         if p.isfile(p.join(path, 'os.py')):
#             return index
#     raise RuntimeError('Could not find standard library path in Python path.')
#
#
# def PythonSysPath(**kwargs):
#     sys_path = kwargs['sys_path']
#
#     dependencies = [p.join(DIR_OF_THIS_SCRIPT, 'python'),
#                     p.join(DIR_OF_THIRD_PARTY, 'requests-futures'),
#                     p.join(DIR_OF_THIRD_PARTY, 'ycmd'),
#                     p.join(DIR_OF_THIRD_PARTY, 'requests_deps', 'idna'),
#                     p.join(DIR_OF_THIRD_PARTY, 'requests_deps', 'chardet'),
#                     p.join(DIR_OF_THIRD_PARTY, 'requests_deps', 'urllib3', 'src'),
#                     p.join(DIR_OF_THIRD_PARTY, 'requests_deps', 'certifi'),
#                     p.join(DIR_OF_THIRD_PARTY, 'requests_deps', 'requests')]
#
#     # The concurrent.futures module is part of the standard library on Python 3.
#     interpreter_path = kwargs['interpreter_path']
#     major_version = int(subprocess.check_output(
#         [interpreter_path, '-c',
#          'import sys; print( sys.version_info[ 0 ] )']).rstrip().decode('utf8'))
#     if major_version == 2:
#         dependencies.append(p.join(DIR_OF_THIRD_PARTY, 'pythonfutures'))
#
#         sys_path[0:0] = dependencies
#         sys_path.insert(GetStandardLibraryIndexInSysPath(sys_path) + 1,
#                         p.join(DIR_OF_THIRD_PARTY, 'python-future', 'src'))
#
#         return sys_path


def Settings(**kwargs):
    return {'interpreter_path': '/pghbio/dbmi/batmanlab/bpollack/conda_envs/mre/bin/python'}
