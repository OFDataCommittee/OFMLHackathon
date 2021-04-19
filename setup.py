import os
import re
import sys
import glob
import sysconfig
import platform
import subprocess
import shutil
import site
from pathlib import Path
import multiprocessing as mp

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

NPROC = mp.cpu_count()

class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        check_prereq("make")
        check_prereq("cmake")
        check_prereq("gcc")
        check_prereq("g++")

        build_directory = os.path.abspath(self.build_temp)
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_directory,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Assuming Makefiles
        build_args += ['--', f'-j{str(NPROC)}']

        self.build_args = build_args

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        env.update(get_build_env(self.build_temp))

        print('-'*10, 'Building C dependencies', '-'*40)
        make_cmd = shutil.which("make")
        setup_path = os.path.abspath(os.path.dirname(__file__))
        shutil.copy(os.path.join(setup_path, "Makefile"),
                    os.path.join(self.build_temp, "Makefile"))
        shutil.copytree(os.path.join(setup_path, "build-scripts"),
                    os.path.join(self.build_temp, "build-scripts"))

        # build dependencies
        subprocess.check_call([f"{make_cmd} deps"],
                              cwd=self.build_temp,
                              shell=True)

        # run cmake prep step
        print('-'*10, 'Running CMake prepare', '-'*40)
        subprocess.check_call(['cmake', setup_path] + cmake_args,
                              cwd=self.build_temp,
                              env=env)


        print('-'*10, 'Building extensions', '-'*40)
        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = Path(self.build_temp).resolve()
        dest = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        dest_path = dest.joinpath("smartredis")
        source_path = build_temp / self.get_ext_filename(ext.name)
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)

def check_prereq(command):
    try:
        out = subprocess.check_output([command, '--version'])
    except OSError:
        raise RuntimeError(
            f"{command} must be installed to build SmartRedis")

def get_build_env(base_path):
    build_env = {}
    hiredis = Path(base_path, "third-party/hiredis/install")
    build_env["HIREDIS_INSTALL_PATH"] = hiredis
    redis_pp = Path(base_path, "third-party/redis-plus-plus/install")
    build_env["REDISPP_INSTALL_PATH"] = redis_pp
    protobuf = Path(base_path, "third-party/protobuf/install")
    build_env["PROTOBUF_INSTALL_PATH"] = protobuf
    pybind = Path(base_path, "third-party/pybind")
    build_env["PYBIND_INSTALL_PATH"] = pybind
    build_env["PYBIND_INCLUDE_PATH"] = pybind.joinpath("include/pybind11/")
    return build_env


ext_modules = [
    CMakeExtension('smartredisPy'),
]

setup(
 # ...
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
