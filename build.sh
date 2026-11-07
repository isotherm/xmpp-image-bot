#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/libfacedetection"
BUILD="$ROOT/build/libfacedetection"
OUT="$ROOT/build"

for command in git cmake c++; do
    command -v "$command" >/dev/null 2>&1 || {
        printf 'Missing build tool: %s\n' "$command" >&2
        exit 1
    }
done

if [[ ! -f "$SRC/CMakeLists.txt" ]]; then
    echo 'Initializing libfacedetection submodule...'
    git -C "$ROOT" submodule update --init --recursive -- libfacedetection
fi
[[ -f "$SRC/CMakeLists.txt" ]] || { echo 'libfacedetection submodule missing' >&2; exit 1; }
[[ -f "$ROOT/face_wrapper.cpp" ]] || { echo 'face_wrapper.cpp missing from repository root' >&2; exit 1; }

mkdir -p "$OUT"
ARCH="$(uname -m)"
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
)
case "$ARCH" in
    arm*|aarch64)
        # Disable x86 SIMD options when supported by the submodule revision.
        CMAKE_ARGS+=( -DENABLE_AVX2=OFF -DENABLE_AVX=OFF -DENABLE_SSE=OFF )
        ;;
esac

printf 'Configuring libfacedetection (%s)...\n' "$ARCH"
cmake -S "$SRC" -B "$BUILD" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD" --parallel "${JOBS:-2}"

# Different upstream revisions/CMake generators may place the .so in
# different subdirectories. Locate the actual shared library.
LIB="$(find "$BUILD" -type l -name 'libfacedetection.so' -print -quit)"
[[ -n "$LIB" ]] || { echo 'Shared libfacedetection.so not found; check CMake build options' >&2; exit 1; }
cp -f "$LIB"* "$OUT/"

printf 'Building ctypes wrapper...\n'
"${CXX:-c++}" -O2 -std=c++11 -fPIC -shared \
    -I "$SRC/src" -I "$BUILD" \
    "$ROOT/face_wrapper.cpp" \
    -L "$OUT" -lfacedetection \
    -Wl,-rpath,'$ORIGIN' \
    -o "$OUT/libfacepython.so"

printf '\nBuild complete:\n  %s\n  %s\n' \
    "$OUT/libfacedetection.so" "$OUT/libfacepython.so"
printf '\nExample:\n  ./process_photo.py photo.jpg --output-dir ./photos --face-library "%s"\n' "$OUT/libfacepython.so"
