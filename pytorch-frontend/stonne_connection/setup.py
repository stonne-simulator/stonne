from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

stonne_src_dir='../../stonne/src'


list_of_src_files_to_link=['torch_stonne.cpp', '../../stonne/stonne_linker_src/stonne_linker.cpp']

#Appending STONNE code to the list in order to link the sources
for filename in os.listdir(stonne_src_dir):
    if((filename != "main.cpp") and (filename.endswith("cpp"))):
        filename_path = os.path.join(stonne_src_dir, filename)
        #print(filename_path)
        list_of_src_files_to_link.append(filename_path)

print(list_of_src_files_to_link)

setup(name='torch_stonne',
      ext_modules=[cpp_extension.CppExtension('torch_stonne', list_of_src_files_to_link, include_dirs=['../../stonne/include', '../../stonne/external'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
