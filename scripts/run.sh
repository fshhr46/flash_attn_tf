
set -euo pipefail
BUILDER_IMAGE_NAME="flash-attn-tf-builder"
BUILDER_CONTAINER_NAME="flash-attn-tf-build-container"
BUILDER_CACHE_VOLUME="flash_attn_tf_builder_cache"
BUILDER_CACHE_DIR="/build_cache"
VENV_DIR="$BUILDER_CACHE_DIR/venv"
BAZEL_CACHE_DIR="$BUILDER_CACHE_DIR/bazel"
PIP_CACHE_DIR="$BUILDER_CACHE_DIR/pip"
ARTIFACTS_DIR="$(pwd)/artifacts"

NO_CLANG_TIDY=true

BAZEL_JOBS_ARG="--jobs=1"

# Additional environment variables above
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=12.4
export TF_CUDA_COMPUTE_CAPABILITIES="7.0,7.5,8.0,8.6,8.9,9.0"

if [ ! -d "$VENV_DIR/bin" ]; then
  echo 'Creating and setting up virtual environment...'
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip3 install --upgrade pip uv
else
  echo 'Using existing virtual environment.'
  source "$VENV_DIR/bin/activate"
fi;

BAZEL_CMD="./bazel --output_user_root=$BAZEL_CACHE_DIR"

./bazel --output_user_root=/build_cache/bazel build --jobs=1 --enable_runfiles build_pip_pkg
uv pip install -r requirements-dev.txt;
./configure.py
[[ "$NO_CLANG_TIDY" == false ]] && \$BAZEL_CMD build $BAZEL_JOBS_ARG //... --config=clang-tidy
\$BAZEL_CMD build $BAZEL_JOBS_ARG --enable_runfiles build_pip_pkg
rm -rf artifacts/*.whl
./bazel-bin/build_pip_pkg artifacts $([ "$RELEASE_VERSION" = true ] && echo --release-version)
echo 'Build complete.'

if [[ "$TESTS" == true ]]; then
  echo 'Running tests...'
  pytest -qq -n auto --cov=flash_attn_tf --cov-report='term-missing' --cov-fail-under=90
  echo 'Tests passed.'
fi