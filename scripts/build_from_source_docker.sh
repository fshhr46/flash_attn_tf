#!/usr/bin/env bash

set -euo pipefail

# change the current working directory to the repository root
cd "$(git rev-parse --show-toplevel)"

BUILDER_IMAGE_NAME="flash-attn-tf-builder"
BUILDER_CONTAINER_NAME="flash-attn-tf-build-container"
BUILDER_CACHE_VOLUME="flash_attn_tf_builder_cache"
BUILDER_CACHE_DIR="/build_cache"
VENV_DIR="$BUILDER_CACHE_DIR/venv"
BAZEL_CACHE_DIR="$BUILDER_CACHE_DIR/bazel"
PIP_CACHE_DIR="$BUILDER_CACHE_DIR/pip"
ARTIFACTS_DIR="$(pwd)/artifacts"

CLEAN=false
MEMORY_LIMIT=""
JOBS=""
NO_CLANG_TIDY=true
RELEASE_VERSION=false
TESTS=false
PUBLISH=false
IS_ON_CI=$([[ "${CI:-}" == "1" ]] && echo true || echo false)

# setup any CI specific things
if [[ "$IS_ON_CI" == true ]]; then
  export PATH="$HOME/.pyenv/bin:$PATH"
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=true
      shift
      ;;
    --no-clang-tidy)
      NO_CLANG_TIDY=true
      shift
      ;;
    --release-version)
      RELEASE_VERSION=true
      shift
      ;;
    --tests)
      TESTS=true
      shift
      ;;
    --memory)
      if [[ -n "${2:-}" ]]; then
        MEMORY_LIMIT="$2"
        shift 2
      else
        echo "Error: --memory requires a value." >&2
        exit 1
      fi
      ;;
    --jobs)
      if [[ -n "${2:-}" ]]; then
        JOBS="$2"
        shift 2
      else
        echo "Error: --jobs requires a value." >&2
        exit 1
      fi
      ;;
    *)
      echo "Error: Invalid argument '$1'. Allowed arguments are '--clean', '--no-clang-tidy', '--release-version', '--tests', '--memory', '--jobs'." >&2
      exit 1
      ;;
  esac
done

# if --clean is passed, removes the docker volume used for caching and also forcibly remove the container if it exists
# before proceeding
if [[ "$CLEAN" == true ]]; then
  echo "Cleaning up Docker volume '$BUILDER_CACHE_VOLUME'..."
  docker ps -a --filter volume="$BUILDER_CACHE_VOLUME" -q | xargs -r docker rm -f
  docker volume rm "$BUILDER_CACHE_VOLUME" || true
fi

# Always attempt to build the Docker image. This ensure we have the latest
# builder image + the code to be built is included at this point.
echo "Attempting to build Docker image '$BUILDER_IMAGE_NAME' from docker/Dockerfile.builder..."
docker build \
  --build-arg USER="$(id -un)" \
  --build-arg UID="$(id -u)" \
  --build-arg GROUP="$(id -gn)" \
  --build-arg GID="$(id -g)" \
  -t "$BUILDER_IMAGE_NAME" \
  -f docker/Dockerfile.builder .
if [ $? -ne 0 ]; then
  echo "Error building Docker image."
  exit 1
fi

# Check and create the cache volume
if ! docker volume inspect "$BUILDER_CACHE_VOLUME" > /dev/null 2>&1; then
  echo "Docker volume '$BUILDER_CACHE_VOLUME' not found. Creating..."
  docker volume create "$BUILDER_CACHE_VOLUME"
  if [ $? -ne 0 ]; then
    echo "Error creating Docker volume '$BUILDER_CACHE_VOLUME'."
    exit 1
  fi
fi

# Create the local artifacts directory if it doesn't exist
mkdir -p "$ARTIFACTS_DIR"

BAZEL_JOBS_ARG=""
if [[ -n "$JOBS" ]]; then
  BAZEL_JOBS_ARG="--jobs=$JOBS"
fi

BUILD_SCRIPT=$(cat <<-EOS
set -euo pipefail

export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=12.4
# A100, L4, H100
export TF_CUDA_COMPUTE_CAPABILITIES="8.9"

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
EOS
)

# Run the Docker container
echo "Running Docker container to build..."
docker run -u $(id -u):$(id -g) --rm --name "$BUILDER_CONTAINER_NAME" \
  ${MEMORY_LIMIT:+--memory="$MEMORY_LIMIT"} \
  -v "$BUILDER_CACHE_VOLUME":/build_cache \
  -v "$ARTIFACTS_DIR":/workspace/artifacts \
  -w /workspace \
  "$BUILDER_IMAGE_NAME" \
  /bin/bash -c "$BUILD_SCRIPT"
if [ $? -ne 0 ]; then
  echo "Error running Docker container."
  exit 1
fi
