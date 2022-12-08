from setuptools import setup
from torch.utils import cpp_extension
from pprint import pprint
import os
import shutil

"""
This file is used to build the Python package that contains the PyTorch extension
corresponding to the STONNE frontend.

NOTE: Due to the implementation of torch.utils.cpp_extension, all files to be compiled
must be in the same directory. Therefore, as a previous step, all source code must be
copied to an auxiliary directory, called 'stonne_src_code', which will be in this folder.
Solution to this problem was found in: https://discuss.pytorch.org/t/pytorch-extension-build-problem/146228

# To install this package:
    python setup.py install
# To uninstall this package:
    pip uninstall torch_stonne
"""


# Current path
PWD = os.path.abspath(".")

# List of root folders which contains the source code
STONNE_SRC_DIRS = [
    os.path.join(PWD,'../../stonne/src')
]

# List of root folders which contains the header/include files
STONNE_INCLUDE_DIRS = [
    os.path.join(PWD, '../../stonne/include/'),
    os.path.join(PWD, '../../stonne/external/')
]

# Name of the auxiliary directory which the source code will be copied
STONNE_SRC_COPY_DIR = os.path.join(PWD,'stonne_src_code')


# List of files that will be compiled (will be filled later)
list_of_src_files_to_link = [
    os.path.abspath('torch_stonne.cpp'),
    os.path.abspath('../../stonne/stonne_linker_src/stonne_linker.cpp')
]
# List of all include dirs (will be filled later)
list_of_include_dirs = []


##################################################
# Copy the source code to an auxiliary directory #
##################################################

# Create the auxiliary directory
os.makedirs(STONNE_SRC_COPY_DIR, exist_ok=True)
# Loop over all files in the source directories recursively
for src_dir in STONNE_SRC_DIRS:
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".cpp"): # only copy .cpp files
                # Original file path
                src_file = os.path.join(root, file)
                # New file path, to distinguish if there are several files with the same name
                # NOTE: if there are two files with the same name in a subfolder with the same name, it won't work
                dst_file = os.path.join(PWD, STONNE_SRC_COPY_DIR, os.path.basename(root) + "_" + file)
                # Copy the file, preserving the file metadata to avoid additional recompiles
                shutil.copy2(src_file, dst_file)


###################################################
#   Get the list of source files to be compiled   #
# and the list of include directories recursively #
###################################################

# Appending STONNE code to the list in order to link the sources
for root, dirs, files in os.walk(STONNE_SRC_COPY_DIR):
    for filename in files:
        if((filename != "main.cpp") and (filename.endswith("cpp"))):
            list_of_src_files_to_link.append(os.path.join(root, filename))

# Appending all STONNE include dirs recursively
for include_dir in STONNE_INCLUDE_DIRS.copy():
    for root, dirs, files in os.walk(include_dir):
        list_of_include_dirs.append(root)

print('# List of source files to be compiled:')
pprint(list_of_src_files_to_link)
print('# List of include directories:')
pprint(list_of_include_dirs)


############################
# Build the Python package #
############################

# See https://pytorch.org/docs/stable/cpp_extension.html for more info

setup(
    # General information about the package
    name='torch_stonne',
    version='1.0.0',
    description='PyTorch extension to interconnect with STONNE frontend',
    url='https://github.com/stonne-simulator/stonne',
    author='Francisco Muñoz Martínez',

    # Build information
    ext_modules=[
        cpp_extension.CppExtension(
            name='torch_stonne',
            sources=list_of_src_files_to_link,
            include_dirs=list_of_include_dirs,
            extra_compile_args=['-O3', '-std=c++17']
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)


#####################
# Clean environment #
#####################

shutil.rmtree(STONNE_SRC_COPY_DIR)