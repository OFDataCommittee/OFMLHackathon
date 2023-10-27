# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import subprocess
import shutil
from pathlib import Path
import multiprocessing as mp

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# get number of processors
NPROC = mp.cpu_count()

class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class CMakeBuild(build_ext):

    @property
    def cmake(self):
        """Find and use installed cmake"""
        cmake_cmd = shutil.which("cmake")
        return cmake_cmd

    @property
    def make(self):
        """Find and use installed cmake"""
        make_cmd = shutil.which("make")
        return make_cmd

    def run(self):
        # Validate dependencies
        check_prereq("cmake")
        check_prereq("make")
        check_prereq("gcc")
        check_prereq("g++")

        # Set up parameters
        source_directory = Path(__file__).parent.resolve()
        build_directory = Path(self.build_temp).resolve()
        cfg = 'Debug' if self.debug else 'Release'

        # Setup build environment
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())

        # Build dependencies
        print('-'*10, 'Building third-party dependencies', '-'*40)
        subprocess.check_call(
            [self.make, "deps"],
            cwd=source_directory,
            shell=False
        )

        # Run CMake config step
        print('-'*10, 'Configuring build', '-'*40)
        config_args = [
            '-S.',
            f'-B{str(build_directory)}',
            '-DSR_BUILD=' + cfg,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(build_directory),
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DSR_PYTHON=ON',
        ]
        subprocess.check_call(
            [self.cmake] + config_args,
            cwd=source_directory,
            env=env
        )

        # Run CMake build step
        print('-'*10, 'Building library', '-'*40)
        build_args = [
            '--build',
            str(build_directory),
            '--',
            f'-j{str(NPROC)}'
        ]
        subprocess.check_call(
            [self.cmake] + build_args,
            cwd=build_directory,
            env=env
        )

        # Move from build temp to final position
        # (Note that we skip the CMake install step because
        # we configured the library to be built directly into the
        # build directory)
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

# check that certain dependencies are installed
def check_prereq(command):
    try:
        _ = subprocess.check_output([command, '--version'])
    except OSError:
        raise RuntimeError(
            f"{command} must be installed to build SmartRedis")

ext_modules = [
    CMakeExtension('smartredisPy'),
]

setup(
 # ... in setup.cfg
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
