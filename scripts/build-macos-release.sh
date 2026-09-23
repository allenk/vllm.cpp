#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "usage: $0 ARTIFACT_ID CHANNEL BUILD_DIR [MLX_ROOT MLX_VERSION MLX_LICENSE]" >&2
  exit 2
fi

artifact_id=$1
channel=$2
build_dir=$3
mlx_root=${4:-}
mlx_version=${5:-}
mlx_license=${6:-}
: "${SOURCE_SHA:?SOURCE_SHA is required}"
: "${VERSION:?VERSION is required}"
: "${EVIDENCE_URL:?EVIDENCE_URL is required}"
: "${SOURCE_DATE_EPOCH:?SOURCE_DATE_EPOCH is required}"

mlx=OFF
if [[ "$artifact_id" == macos-arm64-metal-mlx ]]; then
  mlx=ON
  if [[ -z "$mlx_root" || -z "$mlx_version" || -z "$mlx_license" ]]; then
    echo "MLX artifact requires an exact root, version, and license" >&2
    exit 2
  fi
fi

# TLS: link OpenSSL STATICALLY on this lane, and refuse the lane rather than
# ship a reduced binary if that is impossible.
#
# WHY. On Linux the default `VLLM_CPP_OPENSSL=ON` resolves to the distro's
# `/lib/.../libssl.so.3`, which `scripts/validate-release-archive.py` allows. On
# macOS there is no system OpenSSL to find, so `find_package(OpenSSL)` resolves
# to Homebrew, and the validator's Mach-O allowlist is exactly `/usr/lib/`,
# `/System/Library/`, `@rpath/` and `@loader_path/`
# (`scripts/validate-release-archive.py:462`). MEASURED on macos-15, run
# 35855557148: BOTH macOS lanes built clean and passed every test (112,301 and
# 112,337 assertions) and then failed packaging with
#   forbidden Mach-O install name: /opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib
#   forbidden Mach-O install name: /opt/homebrew/opt/openssl@3/lib/libssl.3.dylib
#
# WHY NOT THE OTHER TWO DISPOSITIONS. `-DVLLM_CPP_BUILD_BORINGSSL=ON` is ruled
# out by the argument `scripts/build-cpu-release.sh:39-43` already makes for the
# musl lane: it fetches from the network at CONFIGURE time, which a release lane
# must not do. `-DVLLM_CPP_HF_DOWNLOAD=OFF` is what the musl lane chose, but musl
# is an `experimental-preview` artifact and `macos-arm64-metal` is a STABLE one,
# so dropping `--model org/repo` there is a regression a release should not make
# on its own authority.
#
# So: static Homebrew OpenSSL, which links the same library with no install name
# at all. If the static archives are absent this lane STOPS with the text below,
# because a green lane that shipped less than it claims is the one outcome worse
# than a red one.
openssl_prefix="$(brew --prefix openssl@3 2>/dev/null || echo /opt/homebrew/opt/openssl@3)"
echo "TLS: openssl@3 prefix = $openssl_prefix"
ls -l "$openssl_prefix/lib" 2>/dev/null || true
if [[ ! -f "$openssl_prefix/lib/libssl.a" || ! -f "$openssl_prefix/lib/libcrypto.a" ]]; then
  echo "TLS: no static libssl.a/libcrypto.a under $openssl_prefix/lib." >&2
  echo "     A dynamic Homebrew link cannot pass the Mach-O install-name gate," >&2
  echo "     and turning VLLM_CPP_HF_DOWNLOAD off would ship a STABLE artifact" >&2
  echo "     that cannot resolve --model org/repo. Decide that explicitly." >&2
  exit 1
fi

cmake -S . -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENSSL_USE_STATIC_LIBS=ON \
  -DOPENSSL_ROOT_DIR="$openssl_prefix" \
  -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_BUILD_VERSION="$VERSION" \
  -DVLLM_CPP_BUILD_EXAMPLES=ON \
  -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA=OFF \
  -DVLLM_CPP_CUDA_ARCHITECTURES= \
  -DVLLM_CPP_HIP=OFF \
  -DVLLM_CPP_HIP_ARCHITECTURES= \
  -DVLLM_CPP_LITERAL_STATIC=OFF \
  -DVLLM_CPP_METAL=ON \
  -DVLLM_CPP_MLX="$mlx" \
  -DMLX_ROOT="$mlx_root" \
  -DVLLM_CPP_TRITON=OFF \
  -DVLLM_CPP_VULKAN=OFF
cmake --build "$build_dir" --target server test_metal_backend -j 2
"$build_dir/tests/test_metal_backend"

release_dir="$build_dir/release"
stage_dir="$release_dir/stage"
metadata_dir="$release_dir/metadata"
archive="$release_dir/vllm.cpp-$VERSION-$artifact_id.tar.gz"
mkdir -p "$release_dir"
python3 scripts/package-server.py --build-dir "$build_dir" --stage-dir "$stage_dir"

compiler=$(c++ --version | head -n 1)
toolchain="$(cmake --version | head -n 1); $(ninja --version)"
c_abi_version=$(sed -n 's/^#define VLLM_ABI_VERSION \([0-9][0-9]*\)$/\1/p' include/vllm.h)
python3 scripts/release_macos_metadata.py \
  --build-dir "$build_dir" \
  --stage-dir "$stage_dir" \
  --output-dir "$metadata_dir" \
  --artifact-id "$artifact_id" \
  --channel "$channel" \
  --version "$VERSION" \
  --c-abi-version "$c_abi_version" \
  --source-commit "$SOURCE_SHA" \
  --source-clean \
  --abi-version "$(sw_vers -productVersion)" \
  --mlx-version "$mlx_version" \
  --mlx-license "$mlx_license" \
  --compiler "$compiler" \
  --toolchain "$toolchain" \
  --evidence-url "$EVIDENCE_URL"
python3 scripts/package-server.py \
  --build-dir "$build_dir" \
  --stage-dir "$stage_dir" \
  --metadata-dir "$metadata_dir" \
  --archive "$archive" \
  --archive-format tar.gz
python3 scripts/validate-release-archive.py \
  --archive "$archive" \
  --archive-format tar.gz \
  --checksum "$archive.sha256" \
  --provenance "$archive.provenance.json" \
  --repo-root . \
  --forbid-path "$PWD/$build_dir"
