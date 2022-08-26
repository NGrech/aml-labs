# aml-labs

Repo for Advanced Machine Learning (DA633E) @MAU part of Computer Science: Applied Data Science, Master's Program

Notebooks are based on [MAU-AML-labs](https://github.com/aeau/MAU-AML-labs/tree/develop/1-computer-vision-lab) notebooks provided for the labs.

## Installation

First install non pyTorch dependencies with [poetry](https://python-poetry.org/) (recommended), for this you will require python 3.8+ and poetry installed.

To install the virtual environment with required packages run:

```bash
poetry install 
```

Then you can install the appropriate version of pyTorch by first running: 

```bash
poetry shell 
```

Then install torch, torchvision and torchaudio using:

For windows with cuda support:

```bash
poe win_cuda
```

For windows without cuda support:

```bash
poe win_cpu
```

For Mac (no cuda support):

```bash
poe mac_cpu
```

For Linux with cuda support:

```bash
poe linux_cuda
```

For Linux without cuda support:

```bash
poe linux_cpu
```

