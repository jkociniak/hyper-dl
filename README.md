# hyper-dl

This is the repository for my research project involving hyperbolic neural networks. Currently in initial phase.

## Structure

First, in `hyper-dl/hypertorch` I keep the implementation of neural networks in PyTorch. It will probably land in separate package, but it is too small for now.

Second, in `hyper-dl/hyperdist` I keep the implementation, models and experiments regarding hyperbolic distance prediction.

## Installation

I use Python 3.9 and did not test anything with other versions.

```
git clone https://github.com/jkociniak/hyper-dl
cd hyper-dl
pip install hypertorch
```

For development, last line should be `pip install -e hypertorch`.
