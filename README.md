# FlashAttention Tensorflow
This repository provides a tensorflow binding to <https://github.com/Dao-AILab/flash-attention>. To avoid depending on pytorch, since torch and tensorflow installations often conflict, this is a fork of the official repo.

Please see [Tri Dao's repo](https://github.com/Dao-AILab/flash-attention) for more information about flash attention. Also check there for how to cite the authors if you used flash attention in your work.

FlashAttention and FlashAttention-2 are free to use and modify (see LICENSE).
Please cite (see below) and credit FlashAttention if you use it.

# Docker builder

```
docker build -t flash-attn-tf-builder \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USER=$(whoami) \
  --build-arg GROUP=$(id -gn) \
  -f docker/Dockerfile.builder .
```

# Building `.whl` from source

1. Make sure your development environment is configured (see above).
2. Run `./scripts/build_from_source_locally.sh --install` to build the project from source and install the package on
   your local Python environment. This is recommended during development.
   
   Alternatively, you can use Docker to avoid issues with local environment and make sure the built `.whl` matches CI by
   running `./scripts/build_from_source_docker.sh`.


## Installation

Requirements:
- CUDA 12.3 and above.
- Linux. Same story as with the pytorch repo. I haven't tested compilation of the tensorflow bindings on windows.
- Tensorflow >= `2.16.*`. The custom call api changed in this version.

To install: `pip install flash-attn-tf` will get the latest release from pypi. This gives you the cuda 12.4
build.

### Installing from source

Flash attention takes a long time to compile unless you have a powerful machine. But if you want to compile from source, I use `cibuildwheel` to compile the releases. You could do the same. Something like (for python 3.12):

```sh
git clone https://github.com/nshepperd/flash_attn_tf
cd flash_attn_tf
cibuildwheel --only cp310-manylinux_x86_64 # I think cibuildwheel needs superuser privileges on some systems because of docker reasons?
```

This will create a wheel in the `wheelhouse` directory. You can then install it with `pip install wheelhouse/flash_attn_tf_*.whl`. Or you could build it without docker using `uv build --wheel`. You need cuda installed in that case.

## Usage

Interface: `src/flash_attn_tf/flash.py`

```py
from flash_attn_tf import flash_mha

# flash_mha : [n, l, h, d] x [n, lk, hk, d] x [n, lk, hk, d] -> [n, l, h, d]
flash_mha(q,k,v,softmax_scale=None, is_causal=False, window_size=(-1,-1))
```

This supports multi-query and grouped-query attention (when hk != h). The `softmax_scale` is the multiplier for the softmax, defaulting to `1/sqrt(d)`. Set `window_size` to positive values for sliding window attention.

### GPU support

FlashAttention-2 currently supports:
1. Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). Support for Turing
   GPUs (T4, RTX 2080) is coming soon, please use FlashAttention 1.x for Turing
   GPUs for now.
2. Datatype fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper GPUs).
3. All head dimensions up to 256. ~~Head dim > 192 backward requires A100/A800 or H100/H800~~. Head dim 256 backward now works on consumer GPUs (if there's no dropout) as of flash-attn 2.5.5.
