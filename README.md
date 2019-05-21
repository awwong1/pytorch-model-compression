# pytorch-model-compression

Model Compression Research Project, Spring/Summer 2019
University of Alberta

Complete rewrite of [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification) to support model compression task.

## Quickstart

```bash
virtualenv venv --python python3
source venv/bin/activate
pip install -r requirements.txt

# Train Alexnet
./cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/alexnet/1
# Profile & Evaluate Alexnet
nvprof --profile-from-start off -o out.prof -- python3 cifar.py -a alexnet --checkpoint checkpoints/cifar10/alexnet/1 --mode evaluate 

```

## License

[MIT](./LICENSE)
