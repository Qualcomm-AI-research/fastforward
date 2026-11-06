#!/usr/bin/env bash
set -euo pipefail

readonly LLAMA_CPP_TAG="b12110"
readonly LLAMA_CPP_ARCHIVE_SHA256="368227edcc229be5f9bed16a2223d79b5e41d0ac2e19bcf5825e82291c060e55"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
INSTALL_ROOT="${REPO_ROOT}/build/llama.cpp/${LLAMA_CPP_TAG}"
readonly INSTALL_ROOT
BIN_DIR="${INSTALL_ROOT}/bin"
readonly BIN_DIR
readonly LLAMA_PERPLEXITY="${BIN_DIR}/llama-perplexity"

if [[ "${1:-}" == "--revision" ]]; then
    printf '%s\n' "${LLAMA_CPP_TAG}"
    exit 0
fi

if [[ -x "${LLAMA_PERPLEXITY}" ]]; then
    "${LLAMA_PERPLEXITY}" --version >&2
    printf '%s\n' "${BIN_DIR}"
    exit 0
fi

for command in curl sha256sum tar; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "${command}" >&2
        exit 1
    fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

archive="${tmpdir}/llama.cpp.tar.gz"
archive_url="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_TAG}/llama-${LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz"

printf 'Downloading llama.cpp %s pre-built binaries\n' "${LLAMA_CPP_TAG}" >&2
curl --fail --location --silent --show-error "${archive_url}" --output "${archive}"
printf '%s  %s\n' "${LLAMA_CPP_ARCHIVE_SHA256}" "${archive}" | sha256sum --check --status

mkdir -p "${INSTALL_ROOT}"
tar --extract --gzip --file "${archive}" --directory "${INSTALL_ROOT}" --strip-components=1

mkdir -p "${BIN_DIR}"
cat > "${LLAMA_PERPLEXITY}" <<'WRAPPER'
#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec env LD_LIBRARY_PATH="${DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "${DIR}/llama-perplexity" "$@"
WRAPPER
chmod +x "${LLAMA_PERPLEXITY}"

"${LLAMA_PERPLEXITY}" --version >&2
printf '%s\n' "${BIN_DIR}"
