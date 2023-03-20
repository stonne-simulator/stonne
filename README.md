# STONNE: A Simulation Tool for Neural Networks Engines

- [STONNE: A Simulation Tool for Neural Networks Engines](#stonne-a-simulation-tool-for-neural-networks-engines)
  - [Bibtex](#bibtex)
  - [Docker image](#docker-image)
  - [What is STONNE](#what-is-stonne)
  - [Design of STONNE](#design-of-stonne)
    - [Flexible DNN Architecture](#flexible-dnn-architecture)
    - [Input Module](#input-module)
    - [Output module](#output-module)
    - [STONNE Mapper: Module for automatic tile generation](#stonne-mapper-module-for-automatic-tile-generation)
  - [Supported Architectures](#supported-architectures)
  - [STONNE User Interface. How to run STONNE quickly.](#stonne-user-interface-how-to-run-stonne-quickly)
    - [Installation](#installation)
    - [How to run STONNE](#how-to-run-stonne)
    - [Help Menu](#help-menu)
    - [Hardware Parameters](#hardware-parameters)
    - [Dimension and tile Parameters](#dimension-and-tile-parameters)
    - [Examples](#examples)
    - [Output](#output)
    - [Generating Energy Numbers](#generating-energy-numbers)
  - [PyTorch Frontend](#pytorch-frontend)
    - [Installation](#installation-1)
    - [Running PyTorch in STONNE](#running-pytorch-in-stonne)
    - [Simulation with real benchmarks](#simulation-with-real-benchmarks)

## Bibtex
Please, if you use STONNE, please cite us:
```
@INPROCEEDINGS{STONNE21,
  author =       {Francisco Mu{\~n}oz-Matr{\'i}nez and Jos{\'e} L. Abell{\'a}n and Manuel E. Acacio and Tushar Krishna},
  title =        {STONNE: Enabling Cycle-Level Microarchitectural Simulation for DNN Inference Accelerators},
  booktitle =    {2021 IEEE International Symposium on Workload Characterization (IISWC)}, 
  year =         {2021},
  volume =       {},
  number =       {},
  pages =        {},
}
```

## Docker image

We have created a docker image for STONNE! Everything is installed in the image so using the simulator is much easier. Also, this image comes with [OMEGA](https://github.com/stonne-simulator/dockerfile) and [SST-STONNE](https://github.com/stonne-simulator/sst-elements-with-stonne) simulators. More information can be found in [our Dockerfile repository](https://github.com/stonne-simulator/dockerfile).

To pull and run the container, just type the following command:

```bash
docker run -it stonnesimulator/stonne-simulators
```

## What is STONNE

The design of specialized architectures for accelerating the inference procedure of Deep Neural Networks (DNNs) is a booming area of research nowadays. While first-generation accelerator proposals used simple fixed dataflows tailored for 
dense DNNs, more recent architectures have argued for flexibility to efficiently support a wide variety of layer types, dimensions, and sparsity. As the complexity of these accelerators grows, it becomes more and more appealing for researchers to have cycle-level simulation tools at their disposal to allow for fast and accurate design-space exploration, and rapid quantification of the efficacy of architectural enhancements during the early stages of a design. To this end, we present STONNE (Simulation TOol of Neural Network Engines), a cycle-level, highly-modular and highly-extensible simulation framework that can plug into any high-level DNN framework as an accelerator device and perform end-to-end evaluation of flexible accelerator microarchitectures with sparsity support, running complete DNN models.

## Design of STONNE

![alt text](https://github.com/francisco-munoz/stonne/blob/master/figures/Top_Level_Stonne_shorter.png)

STONNE is a cycle-level microarchitectural simulator for flexible DNN inference accelerators. To allow for end-to-end evaluations, the simulator is connected with a Deep Learning (DL) framework (Caffe and Pytorch DL frameworks in the current version). Therefore, STONNE can fully execute any dense and sparse DNN models supported by the DL framework that uses as its front-end.
The simulator has been written entirely in C++, following the well-known GRASP and SOLID programming principles of object-oriented design. This has simplified its development and makes it easier the implementation of any kind of DNN inference accelerator microarchitecture, tile configuration mappings and dataflows.

The figure presented above shows a high-level view of STONNE with the three major modules involved in the end-to-end simulation flow:

### Flexible DNN Architecture

This constitutes the principal block of STONNE (see the central block in the figure), and it is mainly composed of the modeled microarchitecture of the flexible DNN accelerator (Simulation Engine) to simulate. The source code of this main block is the foundation of the simulator and is located in the 'stonne/src' folder. This contains different classes for the different components of the architecture as well as the principal class 'STONNEModel' which begins the execution of the simulator. This file contains the main functions to call the simulator and therefore can be view as the 'STONNE API' which is the manner in which the input module can interact with the simulated accelerator, configuring its simulation engine according to the user configuration file, enabling different execution modes such as LFF, loading a certain layer and tile, and transferring the weights and the inputs to the simulator memory.

### Input Module

Due to the flexibility that the \texttt{STONNE API} provides, the simulator can be fed easily using any of the well-known DL frameworks already available. Currently the simulator supports both Caffe and Pytorch DL frameworks and both front-ends with its connections are located in the folders 'pytorch-frontend' and 'caffe-frontend' respectively. Later in this file, we will explain how to install and run every framework on STONNE.

Furthermore, since these DL frameworks require a more complicated installation and use, apart from this mode of execution, we have also enabled the "STONNE User Interface" that facilitates the execution of STONNE. Through this mode, the user is presented with a prompt to load any layer and tile parameters onto a selected instance of the  simulator, and run it with random weights and input values. This allows for faster executions, facilitating rapid prototyping and debugging. This interface can be launched directly in the 'stonne' folder once compiled, and the code is located in 'src/main.cpp' file. Basically, it is a command line that according to hardware and dimensino parameters, allow to run the simulator with random tensors.

### Output module

Once a simulation for a certain layer has been completed, this module is used for reporting simulation statistics such as performance, compute unit utilization, number of accesses to SRAM, wires and FIFOs, etc. Besides, this output module also reports the amount of energy consumed and the on-chip area required by the simulated architecture. These statistics obviously depend on the particular data format (e.g., fp16 or int8) utilized to represent the DNN model's parameters. So, STONNE supports different data formats in the whole end-to-end evaluation process and statistics report. For estimating both area and energy consumption, STONNE utilizes a table-based area and energy model, computing total energy using the cycle-level activity stats for each module. For the current table-based numbers existend in STONNE (see 'stonne/energy\_tables/' path), we ran synthesis using Synopsys Design-Compiler and place-and-route using Cadence Innovus on each module inside the MAERI and SIGMA RTL to populate the table. Users can plug in the numbers for their own implementations as well.

### STONNE Mapper: Module for automatic tile generation

Flexible architectures need to be configured previously to execute any layer, commonly using a tile specification. The specified tile will drastically vary the results obtained in each execution, so choosing a good configuration is an important step when doing simulations. However, finding an optimal configuration is a task that requieres so much time or a deep knowledge of how each mapping would work.

For this reasson, STONNE includes a module called STONNE Mapper. It is a mapper which can automatically generate, given the hardware configuration and the specification of the layer to be simualated, a tile configuration that will be close to the optimum one in most cases. When used, the module will generate an output file with some extra information of the process used to select the tile. STONNE Mapper is an excellent alternative for fast prototyping and a good helper in finding the optimal tile configuration when the search space is very large.

Currently, STONNE Mapper gives support to generate mappings of CONV, FC/DenseGEMM and SparseDense layers (how to use it in every case is explained later). For CONV mappings it integrates and use mRNA, another mapper designed for MAERI presented on ISPASS-2019. For FC/DenseGEMM and SparseDense it implements its own algorithms. In the most of the cases, the generated mappings gets more than the 90% of the performance from optimum configuration (based on the results obtained in our benchmarks), so it has a high trustly degree.

This module and all of the implementations were made in a Final Degree Project in June 2022. Any question about its use or implementation can be made by contacting the author directly ([@Adrian-2105](https://github.com/Adrian-2105)).

## Supported Architectures	

STONNE models all the major components required to build both first-generation rigid accelerators and next-generationflexible DNN accelerators. All the on-chip components are interconnected by using a three-tier network fabric composed of a Distribution Network(DN), a Multiplier Network (MN), and a Reduce Network(RN), inspired by the taxonomy of on-chip communication flows within DNN accelerators. These networks canbe configured to support any topology. Next, we describe the different topologies of the three networks (DN, MN and RN) currently supported in STONNE that are basic building blocks of state-of-the-art accelerators such as the Google’s TPU, Eyeriss-v2, ShDianNao, SCNN, MAERI and SIGMA, among others. These building blocks can also be seen in the figure presented below:

![alt text](https://github.com/francisco-munoz/stonne/blob/master/figures/STONNE_components.png)

## STONNE User Interface. How to run STONNE quickly.

The STONNE User Interface facilitates the execution of STONNE. Through this mode, the user is presented with a prompt to load any layer and tile parameters onto a selected instance of the  simulator, and runs it with random tensors. 


### Installation

The installation of STONNE, along with its user interface,  can be carried out by typing the next commands:
```powershell
cd stonne
make all
```
These commands will generate a binary file `stonne/stonne`. This binary file can be executed to run layers and gemms with any dimensions and any hardware configuration. All the tensors are filled using random numbers. 

### How to run STONNE

Currently, STONNE runs 5 types of operations: Convolution Layers, FC Layers, Dense GEMMs, Sparse GEMMs and SparseDense GEMMs. Please, note that almost any kernel can be, in the end, mapped using these operations. Others operations such as pooling layers will be supported in the future. However, these are the operations that usually dominate the execution time in machine learning applications. Therefore, we believe that they are enough to perform a comprehensive and realistic exploration. Besides, note that a sparse convolution might be also supported as all the convolution layers can be converted into a GEMM operation using the im2col algorithm.

The sintax of a STONNE user interface command to run any of the available operations is as follows:
```powershell
./stonne [-h | -CONV | -FC | -DenseGEMM | -SparseGEMM | SparseDense] [Hardware parameters] [Dimension and tile Parameters]
```

### Help Menu

A help menu will be shown when running the next command:

```powershell
./stonne -h: Obtain further information to run STONNE
```

### Hardware Parameters

The hardware parameters are common for all the kernels. Other parameters can be easily implemented in the simulator. Some parameters are tailored to some specific architectures.


* `num_ms = [x]`
    
    Number of multiplier switches (must be power of 2) (Flexible architecture like MAERI or SIGMA)

* `dn_bw = [x]`
    
    Number of read ports in the SDMemory (must be power of 2) (All architectures)

* `rn_bw = [x]`

    Number of write ports in the SDMemory (must be power of 2) (All architectures)

* `rn_type = [0=ASNETWORK, 1=FENETWORK, 2=TEMPORALRN]`

    Type of the ReduceNetwork to be used (Not supported for SparseGEMM)

* `mn_type = [0=LINEAR, 1=OS_MESH]`
    
    Type of Multiplier network to be used. Linear is for flexible architectures, OS\_MESH for rigid architectures like TPU.

* `mem_ctrl = [MAERI_DENSE_WORKLOAD, SIGMA_SPARSE_GEMM, TPU_OS_DENSE, MAGMA_SPARSE_DENSE]`
    
    Type of memory controller to be used

* `accumulation_buffer = [0,1]`

    Enables the accumulation buffer. Mandatory in Rigid architectures. Also needs to be set to 1 for SparseDense (SpMM) execution.

* `print_stats = [0,1]`

    Flag that enables the printing of the statistics


### Dimension and tile Parameters

Obviously, the dimensions of the kernel depends on the type of the operation that is going to be run.

If you intend to use STONNE Mapper to generate the tile configuration, note that the tile parameters (`T_x`) will be ignored and STONNE will only use the configuration it generates. In the same way, if you use STONNE Mapper, there is not need for the user to manually specify the tile parameters.


Next, it is described the different parameters according to each supported operation:

* **CONV**

    * `layer_name = [CONV]`
        
        Name of the layer to run. The output statistic file will be named accordingly

    * `R = [x]`
        
        Number of filter rows

    * `S = [x]`
    
        Number of filter columns

    * `C =[x]`
    
        Number of filter and input channels

    * `K = [x]`
    
        Number of filters and output channels

    * `G = [x]`
    
        Number of groups

    * `N = [x]`
    
        Number of inputs (Only 1 is supported so far)

    * `X = [x]`
        
        Number of input rows

    * `Y = [x]`
    
        Number of input columns

    * `strides = [x]`
    
        Stride value used in the layer

    * `T_R = [x]`
    
        Number of filter rows mapped at a time

    * `T_S = [x]`
    
        Number of filter columns mapped at a time

    * `T_C = [x]`
    
        Number of filter and input channels per group mapped at a time

    * `T_K = [x]`
    
        Number of filters and output channels per group mapped at a time

    * `T_G = [x]`
    
        Number of groups mapped at a time

    * `T_N = [x]`
    
        Number of inputs mapped at a time (Only 1 is supported so far)

    * `T_X_ = [x]`
    
        Number of input rows mapped at a time

    * `T_Y_ = [x]`
    
        Number of input columns mapped a time

    * **STONNE Mapper**

        * If used, the following parameters can be skipped: `strides`, `T_R`, `T_S`, `T_C`, `T_K`, `T_G`, `T_N`, `T_X_` and `T_Y_`.
        
        * When using it, it is mandatory to also use the option `-accumulation_buffer=1` to ensure that the tile configuration can adjust to the hardware resources.

        * `generate_tile = [0 | none, 1 | performance, 2 | energy, 3 | energy_efficiency]`

          STONNE Mapper is disabled by default (0, `none`). To use it you must to specify a target (1, 2 or 3; also the names can be used). The targets for the tile generation on CONV layers can be: `performance` (1) for maximize the performance, `energy` (2) for minimize the energy consumption and `energy-efficiency` (3) for get a balance between performance and energy.

        * `generator = [Auto, mRNA]`

          [Testing option] At the moment, only `mRNA` algorithm is supported for these type of layers.

    * **Constraints**

        Please make sure that these next constraints are followed (i.e., tile dimension must be multiple of its dimension):
        
        If the architecture to be run is flexible (MAERI or SIGMA):
        1. `T_R % R = 0`
        2. `T_S % S = 0`
        3. `T_C % C = 0`
        4. `T_K % K = 0`
        5. `T_G % G = 0`
        6. `T_X_ % ((X - R + strides) / strides) = 0`
        7. `T_Y_ % ((Y - S + strides) / strides) = 0`

* **FC**

    * `layer_name = [FC]`
    
        Name of the layer to run. The output statistic file will be called by this name

    * `M = [x]`
    
        Number of output neurons

    * `N = [x]`
    
        Batch size

    * `K = [x]`
    
        Number of input neurons

    * `T_M = [x]`
    
        Number of output neurons mapped at a time

    * `T_N = [x]`
    
        Number of batches mapped at a time

    * `T_K = [x]`
    
        Number of input neurons mapped at a time

    * **STONNE Mapper**

        * If used, the following parameters can be skipped: `T_M`, `T_N` and `T_K`.

        * When using it, it is mandatory to also use the option `-accumulation_buffer=1` to ensure that the tile configuration can adjust to the hardware resources.

        * `generate_tile = [0 | none, 1 | performance]`

          STONNE Mapper is disabled by default (0, `none`). To use it you must to specify a target (1; also the names can be used). The only target available at the moment for the tile generation on FC/DenseGEMM layers is `performance` (1) for maximize the performance. However, the generated mapping is also the best mapping for the other targets for this type of layers.

        * `generator = [Auto, StonneMapper, mRNA]`

          [Testing option]  The user can select which algorithm to use to generate the mapping. By default, `StonneMapper` is always used because it gets better results in all cases (because it is a direct improvement of `mRNA`). This option should only be used if it is needed to test the mRNA tile generation for these type of layers.

* **DenseGEMM**

    * `layer_name = [DenseGEMM]`
    
        Name of the layer to run. The output statistic file will be called by this name

    * `M = [x]`
    
        Number of rows MK matrix

    * `N = [x]`
    
        Number of columns KN matrix

    * `K = [x]`
    
        Number of columns MK and rows KN matrix (cluster size)

    * `T_M = [x]`
    
        Number of M rows mapped at a time

    * `T_N = [x]`
    
        Number of N columns at a time

    * `T_K = [x]`
    
        Number of K elements mapped at a time

    * **STONNE Mapper**

        * If used, the following parameters can be skipped: `T_M`, `T_N` and `T_K`.

        * When using it, it is mandatory to also use the option `-accumulation_buffer=1` to ensure that the tile configuration can adjust to the hardware resources.

        * `generate_tile = [0 | none, 1 | performance]`

          STONNE Mapper is disabled by default (0, `none`). To use it you must to specify a target (1; also the names can be used). The only target available at the moment for the tile generation on FC/DenseGEMM layers is `performance` (1) for maximize the performance. However, the generated mapping is also the best mapping for the other targets for this type of layers.

        * `generator = [Auto, StonneMapper, mRNA]`

          [Testing option]  The user can select which algorithm to use to generate the mapping. By default, `StonneMapper` is always used because it gets better results in all cases (because it is a direct improvement of `mRNA`). This option should only be used if it is needed to test the mRNA tile generation for these type of layers.


* **SparseGEMM**

    * `layer_name = [SparseGEMM]`
    
        Name of the layer to run. The output statistic file will be called by this name

    * `M = [x]` 
    
        Number of rows MK matrix

    * `N = [x]`
    
        Number of columns KN matrix

    * `K = [x]`
    
        Number of columns MK and rows KN matrix (cluster size)

    * `MK_sparsity = [x]` 
    
        Percentage of sparsity MK matrix (0-100)

    * `KN_sparsity = [x]`
    
        Percentage of sparsity KN matrix (0-100)

    * `dataflow = [MK_STA_KN_STR, MK_STR_KN_STA]`
    
        Dataflow to use during operations 

    * `optimize = [0,1]`
    
        Apply compiler-based optimizations


* **SparseDense**

    * `layer_name = [SparseDense]`
    
        Name of the layer to run. The output statistic file will be called by this name

    * `M = [x]` 
    
        Number of rows MK matrix

    * `N = [x]`
    
        Number of columns KN matrix

    * `K = [x]`
    
        Number of columns MK and rows KN matrix (cluster size)

    * `MK_sparsity = [x]` 
    
        Percentage of sparsity MK matrix (0-100)

    * `T_N = [x]`
    
        Number of N columns mapped at a time

    * `T_K = [x]`
    
        Number of K elements mapped at a time

    * **STONNE Mapper**

        * If used, the following parameters can be skipped: `T_N` and `T_K`.
        
        * When using it, it is mandatory to also use the option `-accumulation_buffer=1` to ensure that the tile configuration can adjust to the hardware resources.

        * `generate_tile = [0 | none, 1 | performance]`

          STONNE Mapper is disabled by default (0, `none`). To use it you must to specify a target (1; also the names can be used). The only target available at the moment for the tile generation on FC/DenseGEMM layers is `performance` (1) for maximize the performance. However, the generated mapping is also the best mapping for the other targets for this type of layers.

        * `generator = [Auto, StonneMapper]`

          [Testing option] At the moment, only `StonneMapper` algorithm is supported for these type of layers.

### Examples

Example running a CONV layer (manual mapping): 
```powershell
./stonne -CONV -R=3 -S=3 -C=6 -G=1 -K=6 -N=1 -X=20 -Y=20 -T_R=3 -T_S=3 -T_C=1 -T_G=1 -T_K=1 -T_N=1 -T_X_=3 -T_Y_=1 -num_ms=64 -dn_bw=8 -rn_bw=8
```

Example running a CONV layer generating the tile with STONNE Mapper (energy target): 
```powershell
./stonne -CONV -R=3 -S=3 -C=6 -G=1 -K=6 -N=1 -X=20 -Y=20 -generate_tile=energy -num_ms=64 -dn_bw=8 -rn_bw=8 -accumulation_buffer=1
```

Example running a FC layer (manual mapping):
```powershell
./stonne -FC -M=20 -N=20 -K=256 -num_ms=256 -dn_bw=64 -rn_bw=64 -T_K=64 -T_M=2 -T_N=1
```

Example running a FC layer generating the tile with STONNE Mapper (with mRNA algorithm and performance target):
```powershell
./stonne -FC -M=20 -N=20 -K=256 -generate_tile=performance -generator=mRNA -num_ms=256 -dn_bw=64 -rn_bw=64 -accumulation_buffer=1
```

Example of running a DenseGEMM (manual mapping):
```powershell
/stonne -DenseGEMM -M=20 -N=20 -K=256 -num_ms=256 -dn_bw=64 -rn_bw=64 -T_K=64 -T_M=2 -T_N=1
```

Example of running a DenseGEMM over TPU:
```powershell
./stonne -DenseGEMM -M=4 -N=4 -K=16 -ms_rows=4 -ms_cols=4 -dn_bw=8 -rn_bw=16  -T_N=4 -T_M=1 -T_K=1 -accumulation_buffer=1 -rn_type="TEMPORALRN" -mn_type="OS_MESH" -mem_ctrl="TPU_OS_DENSE"
```

Example of running a SparseGEMM:
```powershell
./stonne -SparseGEMM -M=20 -N=20 -K=256 -num_ms=128 -dn_bw=64 -rn_bw=64  -MK_sparsity=80 -KN_sparsity=10 -dataflow=MK_STA_KN_STR
```

Example of running a SparseDense (manual mapping):
```powershell
./stonne -SparseDense -M=20 -N=20 -K=256 -MK_sparsity=80 -T_N=4 -T_K=32 -num_ms=128 -dn_bw=64 -rn_bw=64 -accumulation_buffer=1
```
Note that accumulation buffer needs to be set to 1 for the SparseDense case to work

Example of running a SparseDense generating the tile with STONNE Mapper (performance [1] target):
```powershell
./stonne -SparseDense -M=20 -N=20 -K=256 -MK_sparsity=80 -generate_tile=1 -num_ms=128 -dn_bw=64 -rn_bw=64 -accumulation_buffer=1
```

### Output

Every layer execution generates three files in the path in which the simulator has been executed (the env variable OUTPUT_DIR can be set to indicate another output path): 

- A JSON file with all the hardware statistics generated during the execution. 

- A counters file with the number of use of every component of the architecture generated. This can be utilized to generate the energy model.

- [Only if STONNE Mapper was used] A brief report about the process made by the module to select an efficient tile and the tile used.

Note that after the execution, the results obtained in the output tensor by the simulator are compared with a CPU algorithm to ensure the correctness of the simulator. Note that if the simulator does not output the correct results, an assertion will raise at the end of the execution. 



### Generating Energy Numbers

In order to generate the energy consumption of the execution we have developed a Python script that takes in the counters file generated during the execution and a table-based energy model. The script is located in energy_tables folder and can be run by means of the next command:

```powershell
./calculate_energy.py [-v] -table_file=<Energy numbers file> -counter_file=<Runtime counters file> -[out_file=<output file>]
```
The current energy numbers are located in the file energy_tables/energy_model.txt. We obtained the energy numbers through synthesis using Synopsys Design-Compiler and place-and-route using Cadence Innovus on each module inside the MAERI and SIGMA RTL to populate the table. Users can plug in the numbers for their own implementations as well.

## PyTorch Frontend

At this point, the user must be familiar with the usage of STONNE and the set of statistics that the tool is able to output. However, with the STONNE user interface presented previously, the user must have realized that the inputs and outputs coming in and out in the simulator are random. Here, it is explained how to run real DNN models using pytorch and STONNE as a computing device. 

The pytorch-frontend is located in the folder 'pytorch-frontend' and this basically contains the Pytorch official code Version 1.7 with some extra files to create the simulation operations and link them with the 'stonne/src' code. The current version of the frontend is so well-organized that running a pytorch DNN model on STONNE is straightforward. 

### Installation

First, you will need Python 3.6 or higher and a C++14 compiler. Also, we highly recommend to create an Anaconda environment before continue. Learn how to install it in their [official documentation](https://docs.anaconda.com/anaconda/install/index.html). Once installed, you can create and activate a new virtual environment with the next commands:

```powershell
conda create --name stonne python=3.8 
conda activate stonne
```

Once the environment is ready, you have to install some installation dependencies before continue. You can install them with the next commands:

```powershell
pip install pyyaml==6.0 setuptools==45.2.0 numpy==1.23.5 
```

Now you can start to install the PyTorch frontend. First, if you don't have CUDA on your computer, export the next variables:

```powershell
export MAX_JOBS=1
export NO_CUDA=YES
export NO_CUDA=1
```

Second, you can build and install PyTorch (`torch`) from sources using the next commands (it takes around 20 minutes):

```powershell
cd pytorch-frontend/
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

Third, you can build and install our PyTorch frontend (`torch_stonne`) package using the next commands:

```powershell
cd stonne_connection/
python setup.py install
```

Finally, to be able to run all the benchmarks, you will need to install some extra dependencies. We recommend you to install the specific versions listed below in order to avoid package dependency problems and overwriting the previous torch installation. You can install them with the next commands:

```powershell
pip install transformers==4.25.1
pip install torchvision==0.8.2 --no-deps
```

You can check that the installation was successful by running the next test simulation:

```powershell
python $STONNE_ROOT/pytorch-frontend/stonne_connection/test_simulation.py
```

### Running PyTorch in STONNE

Running pytorch using STONNE as a device is almost straightforward.
Let's assume we define a DNN model using PyTorch. This model is composed of a single and simple convolutional layer. Next, we present this code:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5,5,5, groups=1) # in_channels=5, out_channels=5, filter_size=5
    def forward(self, x):
        x = self.conv1(x)
        return x
```

This code can be easily run in CPU just by means of creating an object of type Net and running the forward method with the correct tensor shape as input.

```python
net = Net()
print(net)
input_test = torch.randn(5,50,50).view(-1,5,50,50)
result  = net(input_test)
```

Migrating this model to STONNE is as simple as turning the Conv2d operation into a SimulatedConv2d operation. Next, we can observe an example:
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.SimulatedConv2d(5,5,5,'$PATH_TO_STONNE/simulation_files/maeri_128mses_128_bw.cfg', 'dogsandcats_tile.txt', sparsity_ratio=0.0, stats_path='.', groups=1) 
    def forward(self, x):
        x = self.conv1(x)
        return x
```

As we can see, we have inserted 4 new parameters:

* `sim_file (str)`

    This is the path to the configuration file of STONNE. This file defines the hardware to be simulated in every execution of STONNE. You can see multiple examples in the folder 'simulation_files'.

* `tile (str)`

    This is the path to a file that defines the tile to be used to partition that layer. An example of this file might be found in `minibenchmarks/dogsandcats_tile.txt` (note that an example for a linear tile file might be found in `minibenchmarks/dogsandcats_tile_fc.txt`). Also an example using STONNE Mapper for generate automatically a tile can be found in `minibenchmarks/dogsandcats_tile_stonnemapper.txt` (same parameters as if used from the CLI). This parameter only will make sense if the hardware configuration file contains a dense memory controller. If the memory controller is sparse, then the execution will not require tiling as it is explained in SIGMA paper.

* `sparsity_ratio (float 0.0-1.0)`

    This is the sparsity ratio used to prune the weight tensor. This parameter only makes sense if a sparsity controller is used in the hardware configuration file. Otherwise this will be ignored.  They way to proceed in the current version of STONNE is indicating this parameter. Then, previously to the simulation, the weight tensor is pruned accordingly to that parameter and the bitmaps are created accordingly. Note that the weights are not retrained and therefore this will affect to the accuracy of the model. However, in terms of a simulation perspective, this lower accuracy is not affected at all. Obviously, this is a way to proceed. It is possible, with low efforts, to run an already pruned and re-trained model. To do so, the code have to be briefly modified to remove the pruning functions and use the real values as they are. By the moment, STONNE only allows bitmap representation of sparsity. If you have a model with other compression format, you could either code your own memory controller to support it or code a simple function to turn your representation format into a bitmap representation. 

* `stats_path`

    This is an optional parameter and points to a folder in which the stats of the simulation of that layer will be stored. 


The addition of these 4 parameters and the modification of the function will let PyTorch run the layer in STONNE obtaining the real tensors.

In the current version of the pytorch-frontend we also support `nn.SimulatedLinear` and `torch_stonne.SimulatedMatmul` operations that correspond with both `nn.Linear` and `nn.Matmul` operations in the original PyTorch framework. The only need is to change the name of the functions and indicate the 3 extra parameters. We still do not support SparseDense operations on the pytorch-frontend.


### Simulation with real benchmarks

In order to reduce the effort of the user, we have already migrated some models to STONNE. By the moment, we have 4 DNN benchmarks in this framework: Alexnet, SSD-mobilenets, SSD-Resnets1.5 and BERT. All of them are in the folder 'benchmarks'. Note that to migrate these models, we have had to understand the code of all of them, locate the main kernels (i.e., convolutions, linear and matrix multiplication operations) and turn the functions into the simulated version. That is the effort you require to migrate a new model. We will update this list over time. 

Running these models is straightforward as we have prepared a script (`benchmarks/run_benchmarks.py` file) that performs all the task automatically. Next, we present one example for each network:

```
cd benchmarks
```

- Running BERT:  
```
python run_benchmarks.py "bert" "../simulation_files/sigma_128mses_64_bw.cfg" "NLP/BERT/tiles/128_mses/" "0.0" ""
```

- Running SSD-Mobilenets
```
python run_benchmarks.py "ssd_mobilenets" "../simulation_files/sigma_128mses_64_bw.cfg" "object_detection/ssd-mobilenets/tiles/128_mses" "0.0" ""
```

- Running SSD-Resnets:
```
 python run_benchmarks.py "ssd_resnets" "../simulation_files/sigma_128mses_64_bw.cfg" "object_detection/ssd-mobilenets/tiles/128_mses" "0.0" "" 
```
