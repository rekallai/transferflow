
import sys, os
try:
    from setuptools import setup
    from setuptools import find_packages
    from setuptools.command.install import install as _install
    from setuptools.command.sdist import sdist as _sdist
except ImportError:
    from distutils.core import setup
    from distutils.core import find_packages
    from distutils.command.install import install as _install
    from distutils.command.sdist import sdist as _sdist

def _run_make(dir):
    from subprocess import call
    call(['make'],
         cwd=os.path.join(dir, 'transferflow/object_detection/utils'))
    call(['make', 'hungarian'],
         cwd=os.path.join(dir, 'transferflow/object_detection/utils'))


class install(_install):
    def run(self):
        _install.run(self)
        self.execute(_run_make, (self.install_lib,),
                     msg="Build Non-Python dependencies")

setup(
    name="transferflow",
    version="0.1.8",
    description='Transfer learning for Tensorflow',
    url='https://github.com/dominiek/transferflow',
    cmdclass={'install': install},
    include_package_data=True,
    install_requires=[
        'nnpack>=0.1.0',
        'numpy>=1.12.0',
        'opencv-python>=3.2.0.6',
        'pytest>=3.0.6',
        'scipy>=0.18.1',
        'tensorflow>=0.12.1'
    ],
    packages=find_packages()
)
