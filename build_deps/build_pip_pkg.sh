#!/usr/bin/env bash

set -ex

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function is_linux() {
  [[ "${PLATFORM}" == "linux" ]]
}

if ! ( is_linux ); then
  echo "Error: This script must be run on Linux only."
  exit 1
fi

PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/flash_attn_tf/"

function abspath() {
  cd "$(dirname $1)"
  echo "$PWD/$(basename $1)"
  cd "$OLDPWD"
}

function main() {
  DEST=${1}
  RELEASE_FLAG=${2}

  if [[ -z ${DEST} ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  mkdir -p ${DEST}
  DEST=$(abspath "${DEST}")
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
  echo $(date) : "=== Using tmpdir: ${TMPDIR}"
  echo "=== Copy TensorFlow Snap Addons files"

  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}requirements.txt "${TMPDIR}"
  touch ${TMPDIR}/stub.cc

  rsync -avm -L --exclude='*_test.py' ${PIP_FILE_PREFIX} "${TMPDIR}"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  BUILD_CMD="setup.py bdist_wheel"

  if [[ -z ${RELEASE_FLAG} ]]; then
    python3 ${BUILD_CMD}
  else
    python3 ${BUILD_CMD} ${RELEASE_FLAG}
  fi

  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
