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

# Calculate number of floating point operations
nvprof --metrics flop_count_dp,flop_count_sp,flop_count_hp,flop_count_sp_special --export-profile out.prof --force-overwrite --print-summary -- python3 cifar.py -a alexnet --checkpoint checkpoints/cifar10/alexnet/1 --mode profile
```

## nvprof Metrics
* **flop_count_dp**
  * Number of double-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.
* **flop_count_sp**
  * Number of single-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count. The count does not include special operations.
* **flop_count_hp**
  * Number of half-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.
* **flop_count_sp_special**
  * Number of single-precision floating-point special operations executed by non-predicated threads.

## License

[MIT](./LICENSE)
