#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

# c++ compiling and pybind setup code copied from https://github.com/pybind/python_example.git

# License:

# Copyright (c) 2016 The Pybind Development Team, All rights reserved.
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
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to the author of this software, without
# imposing a separate written license agreement for such Enhancements, then you
# hereby grant the following license: a non-exclusive, royalty-free perpetual
# license to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such enhancements or
# derivative works thereof, in binary and source code form.


"""The setup script."""
import re

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils import ccompiler
import sys
import setuptools


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['clasp', 'numpy', 'scipy', 'pywavelets', 'matplotlib',
                'skyfield', 'clipt', 'pybind11']

setup_requirements = []

test_requirements = ['pytest', 'hdrstats']

data_files = []
package_data = {}

radiance_compile_args = ["-O2", "-DBSD", "-DNOSTEREO", "-Dfreebsd"]
radiance_include = ['ray/src/rt', 'ray/src/common']
lib_dir = 'src/lib'
srcdir = 'src/'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()


radiancelib = [
    "Version.c",
    "ray/src/common/paths.c",
    "ray/src/rt/ambcomp.c",
    "ray/src/rt/ambient.c",
    "ray/src/rt/ambio.c",
    "ray/src/rt/aniso.c",
    "ray/src/rt/ashikhmin.c",
    "ray/src/rt/data.c",
    "ray/src/rt/dielectric.c",
    "ray/src/rt/fprism.c",
    "ray/src/rt/freeobjmem.c",
    "ray/src/rt/func.c",
    "ray/src/rt/glass.c",
    "ray/src/rt/initotypes.c",
    "ray/src/rt/m_alias.c",
    "ray/src/rt/m_brdf.c",
    "ray/src/rt/m_bsdf.c",
    "ray/src/rt/m_clip.c",
    "ray/src/rt/m_direct.c",
    "ray/src/rt/m_mirror.c",
    "ray/src/rt/m_mist.c",
    "ray/src/rt/mx_data.c",
    "ray/src/rt/mx_func.c",
    "ray/src/rt/noise3.c",
    "ray/src/rt/normal.c",
    "ray/src/rt/o_cone.c",
    "ray/src/rt/o_face.c",
    "ray/src/rt/o_instance.c",
    "ray/src/rt/o_mesh.c",
    "ray/src/rt/p_data.c",
    "ray/src/rt/p_func.c",
    "ray/src/rt/pmap.c",
    "ray/src/rt/pmapamb.c",
    "ray/src/rt/pmapbias.c",
    "ray/src/rt/pmapcontrib.c",
    "ray/src/rt/pmapdata.c",
    "ray/src/rt/pmapdiag.c",
    "ray/src/rt/pmapio.c",
    "ray/src/rt/pmapmat.c",
    "ray/src/rt/pmapopt.c",
    "ray/src/rt/pmapparm.c",
    "ray/src/rt/pmaprand.c",
    "ray/src/rt/pmapray.c",
    "ray/src/rt/pmapsrc.c",
    "ray/src/rt/pmaptype.c",
    "ray/src/rt/pmcontrib2.c",
    "ray/src/rt/pmutil.c",
    "ray/src/rt/preload.c",
    "ray/src/rt/raytrace.c",
    "ray/src/rt/renderopts.c",
    "ray/src/rt/source.c",
    "ray/src/rt/sphere.c",
    "ray/src/rt/srcobstr.c",
    "ray/src/rt/srcsamp.c",
    "ray/src/rt/srcsupp.c",
    "ray/src/rt/t_data.c",
    "ray/src/rt/t_func.c",
    "ray/src/rt/text.c",
    "ray/src/rt/virtuals.c"
    ]

rtradlib = [
    "ray/src/common/addobjnotify.c",
    "ray/src/common/badarg.c",
    "ray/src/common/biggerlib.c",
    "ray/src/common/bmalloc.c",
    "ray/src/common/bmpfile.c",
    "ray/src/common/bsdf.c",
    "ray/src/common/bsdf_m.c",
    "ray/src/common/bsdf_t.c",
    "ray/src/common/byteswap.c",
    "ray/src/common/caldefn.c",
    "ray/src/common/calexpr.c",
    "ray/src/common/calfunc.c",
    "ray/src/common/calprnt.c",
    "ray/src/common/ccolor.c",
    "ray/src/common/ccyrgb.c",
    "ray/src/common/chanvalue.c",
    "ray/src/common/clip.c",
    "ray/src/common/color.c",
    "ray/src/common/colrops.c",
    "ray/src/common/cone.c",
    "ray/src/common/cvtcmd.c",
    "ray/src/common/depthcodec.c",
    "ray/src/common/dircode.c",
    "ray/src/common/disk2square.c",
    "ray/src/common/ealloc.c",
    "ray/src/common/eputs.c",
    "ray/src/common/erf.c",
    "ray/src/common/error.c",
    "ray/src/common/expandarg.c",
    "ray/src/common/ezxml.c",
    "ray/src/common/face.c",
    "ray/src/common/falsecolor.c",
    "ray/src/common/fdate.c",
    "ray/src/common/fgetline.c",
    "ray/src/common/fgetval.c",
    "ray/src/common/fgetword.c",
    "ray/src/common/fixargv0.c",
    "ray/src/common/fltdepth.c",
    "ray/src/common/font.c",
    "ray/src/common/fputword.c",
    "ray/src/common/free_os.c",
    "ray/src/common/fropen.c",
    "ray/src/common/fvect.c",
    "ray/src/common/gethomedir.c",
    "ray/src/common/getlibpath.c",
    "ray/src/common/getpath.c",
    "ray/src/common/header.c",
    "ray/src/common/hilbert.c",
    "ray/src/common/idmap.c",
    "ray/src/common/image.c",
    "ray/src/common/instance.c",
    "ray/src/common/interp2d.c",
    "ray/src/common/invmat4.c",
    "ray/src/common/lamps.c",
    "ray/src/common/linregr.c",
    "ray/src/common/loadbsdf.c",
    "ray/src/common/loadvars.c",
    "ray/src/common/lookup.c",
    "ray/src/common/mat4.c",
    "ray/src/common/mesh.c",
    "ray/src/common/modobject.c",
    "ray/src/common/multisamp.c",
    "ray/src/common/myhostname.c",
    "ray/src/common/normcodec.c",
    "ray/src/common/objset.c",
    "ray/src/common/octree.c",
    "ray/src/common/otypes.c",
    "ray/src/common/paths.c",
    "ray/src/common/plocate.c",
    "ray/src/common/portio.c",
    "ray/src/common/process.c",
    "ray/src/common/quit.c",
    "ray/src/common/readfargs.c",
    "ray/src/common/readmesh.c",
    "ray/src/common/readobj.c",
    "ray/src/common/readoct.c",
    "ray/src/common/resolu.c",
    "ray/src/common/rexpr.c",
    "ray/src/common/savestr.c",
    "ray/src/common/savqstr.c",
    "ray/src/common/sceneio.c",
    "ray/src/common/spec_rgb.c",
    "ray/src/common/tcos.c",
    "ray/src/common/timegm.c",
    "ray/src/common/tmap16bit.c",
    "ray/src/common/tmapcolrs.c",
    "ray/src/common/tmapluv.c",
    "ray/src/common/tmaptiff.c",
    "ray/src/common/tmesh.c",
    "ray/src/common/tonemap.c",
    "ray/src/common/triangulate.c",
    "ray/src/common/urand.c",
    "ray/src/common/urind.c",
    "ray/src/common/wordfile.c",
    "ray/src/common/words.c",
    "ray/src/common/wputs.c",
    "ray/src/common/xf.c",
    "ray/src/common/zeroes.c",
    "ray/src/common/unix_process.c"
    ]

libs = {
    'rcraycalls': ['rcraycalls.c', 'ray/src/rt/raypcalls.c',
                   'ray/src/rt/rayfifo.c'],
    'raycalls': ['ray/src/rt/raycalls.c', 'ray/src/rt/raypcalls.c',
                 'ray/src/rt/rayfifo.c'],
    'rcontribcfiles': ['rcinit.c', 'rcontribparts.c', 'rc3.c',
                       '/ray/src/rt/rc2.c'],
    'rtracecfiles': ['rtinit.c', 'rtraceparts.c', 'ray/src/rt/duphead.c',
                     'ray/src/rt/persist.c', 'ray/src/rt/source.c',
                     'ray/src/rt/pmapray.c'],
    'radiance': radiancelib,
    'rtrad': rtradlib
    }

rcontrib_c_files = ['render.cpp', 'rcontrib.cpp']
rtrace_c_files = ['render.cpp', 'rtrace.cpp']

radiance_include = [f'{srcdir}{i}' for i in radiance_include]

for ke in libs:
    libs[ke] = [f'{srcdir}{i}' for i in libs[ke]]

ext_modules = [
    Extension(
        'raytraverse.craytraverse',
        ['src/pyhelpers.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include()
            ],
        language='c++'
        ),
    Extension(
        'raytraverse.crenderer.rcontrib_c',
        [f'{srcdir}{i}' for i in rcontrib_c_files],
        include_dirs=[*radiance_include, get_pybind_include()],
        libraries=['rcraycalls', 'radiance', 'rtrad', 'rcontribcfiles'],
        depends=[],
        library_dirs=[lib_dir],
        language='c++'
        ),
    Extension(
        'raytraverse.crenderer.rtrace_c',
        [f'{srcdir}{i}' for i in rtrace_c_files],
        include_dirs=[*radiance_include, get_pybind_include()],
        libraries=['raycalls', 'radiance', 'rtrad', 'rtracecfiles'],
        library_dirs=[lib_dir],
        language='c++'
        )
]


def compile_c_libraries():
    rcompiler = ccompiler.new_compiler()
    for rinc in radiance_include:
        rcompiler.add_include_dir(rinc)
    for k, v in libs.items():
        rcompiler.compile(v, extra_preargs=radiance_compile_args)
        o = rcompiler.object_filenames(v)
        rcompiler.create_static_lib(o, k, output_dir=lib_dir, target_lang='c++')
    return {k: rcompiler.library_filename(k) for k in libs}


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        # self.compile_static_libraries()
        dependencies = compile_c_libraries()
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO',
                                  '"{}"'.format(self.distribution.
                                                get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
            ext.dependencies = [v for k, v in dependencies.items()
                                if k in ext.libraries]

        build_ext.build_extensions(self)


setup(
    author="Stephen Wasilewski",
    author_email='stephanwaz@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="variance based adaptive sampling for CBDM",
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'raytraverse=raytraverse.cli:main',
            'genskyvec_sc=raytraverse.gsv:main'
        ],
    },
    install_requires=requirements,
    license="Mozilla Public License 2.0 (MPL 2.0)",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='raytraverse',
    name='raytraverse',
    packages=find_packages(),
    data_files=data_files,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    package_data=package_data,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.enterpriselab.ch/lightfields/raytraverse',
    version='0.1.0',
    zip_safe=True,
    )
