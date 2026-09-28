# Builds glm.js from a fresh checkout of the repository and produces a slim
# runtime image.
#
# Stage 1 (builder): CUDA devel toolchain compiles the sm_120a kernels (nvcc),
# links the native node addon (node-gyp), and compiles TypeScript (tsc).
# Stage 2 (runtime): only CUDA runtime libraries, Node.js, and the built tree.
#
#   docker build -t glmjs .
#   docker run --gpus all -it glmjs
#
# Toolchain choices mirror the development host: CUDA 13.3 on Ubuntu 24.04,
# gcc 13, Node 24.

FROM nvidia/cuda:13.3.1-devel-ubuntu24.04 AS builder

ARG REPO_URL=https://github.com/koush/slopllm.git
ARG REPO_REF=main
ARG NODE_VERSION=v24.16.0

ENV DEBIAN_FRONTEND=noninteractive

# libnccl-dev/libnccl2 already ship in the devel image (held by NVIDIA to the
# version validated for CUDA 13.3; asking apt for them would try to upgrade
# the held packages and fail). python3 is required by node-gyp.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        make \
        g++ \
        python3 \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Node.js, pinned to the version used by the development toolchain.
RUN curl -fsSL "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz" \
    | tar -xJ -C /usr/local --strip-components=1

# The flashinfer submodule is recorded with an SSH remote; rewrite it to HTTPS
# so recursive clones work without credentials inside the container.
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

WORKDIR /build

# Clone the superproject, then fetch submodules fully so their pinned
# commits (e.g. flashinfer -> 3rdparty/cccl) are always available.
#
# REPO_REF may be a branch name or a commit SHA. BuildKit caches this layer
# by its arguments, so a rebuild with an unchanged REPO_REF (e.g. the
# default "main") can reuse a cached clone that predates newer commits.
# To build from fresh source, pass the current remote SHA:
#   REPO_REF=$(git rev-parse origin/main) docker compose build
RUN git clone "${REPO_URL}" glm.js \
    && git -C glm.js checkout "${REPO_REF}" \
    && git -C glm.js submodule update --init --recursive --jobs 8

WORKDIR /build/glm.js

# Install exactly what package-lock.json resolves (includes the vendored
# @huggingface/transformers package via symlink into vendor/). npm also
# builds the CUDA kernels and native addon here because the root project
# has a binding.gyp (node-gyp rebuild runs during ci).
RUN npm ci && npm cache clean --force

# Generate the gitignored TypeScript declarations for the vendored
# @huggingface/transformers package: its ./tokenizers export maps the "types"
# condition into types/, which the fork excludes from git, so a fresh checkout
# must regenerate them. This needs the fork's own dev dependencies (pinned
# tsc 5.9.3, @webgpu/types). tsc --build reports pre-existing type errors in
# the fork's src but still emits declarations; the test below gates on the
# declarations actually being produced.
RUN cd vendor/transformers.js/packages/transformers \
    && npm install --no-save --ignore-scripts \
    && { node_modules/.bin/tsc --build || echo "typegen: known fork src errors; declarations still emitted"; } \
    && test -f types/models/auto/tokenization_auto.d.ts

# Builds the TypeScript output (make/node-gyp output is already complete from
# the npm ci step above, so this is mostly incremental).
RUN npm run build:all

# Slim the tree down to runtime needs:
# - git history and build-only vendors (flashinfer, huggingface.js)
# - onnxruntime + sharp: these ship only as deps of the vendored transformers
#   package, and the project imports just its ./tokenizers entry, whose module
#   graph resolves nothing beyond @huggingface/{jinja,tokenizers}
# - dev toolchains (typescript, @types) and the fork's typegen node_modules
# - object files; only glm.node and libglm_ops.so are needed at runtime
# - everything in the transformers.js monorepo except packages/transformers
RUN rm -rf .git \
        vendor/flashinfer \
        vendor/huggingface.js \
        tests \
        dist/tests \
        node_modules/onnxruntime-node \
        node_modules/onnxruntime-web \
        node_modules/onnxruntime-common \
        node_modules/sharp \
        node_modules/@img \
        node_modules/typescript \
        node_modules/@types \
        vendor/transformers.js/packages/transformers/node_modules \
        vendor/transformers.js/packages/transformers/docs \
        vendor/transformers.js/packages/transformers/tests \
        vendor/transformers.js/packages/transformers/scripts \
        vendor/transformers.js/packages/transformers/jest.config.mjs \
        vendor/transformers.js/packages/transformers/tsconfig.tsbuildinfo \
    && mv build/Release/glm.node build/Release/libglm_ops.so /tmp/ \
    && rm -rf build \
    && mkdir -p build/Release \
    && mv /tmp/glm.node /tmp/libglm_ops.so build/Release/ \
    # prune the transformers.js monorepo down to the one vendored package
    && find vendor/transformers.js -mindepth 1 -maxdepth 1 \
           ! -name packages -exec rm -rf {} + \
    && find vendor/transformers.js/packages -mindepth 1 -maxdepth 1 \
           ! -name transformers -exec rm -rf {} + \
    && find node_modules/.bin -xtype l -delete

# ---------------------------------------------------------------------------
# Runtime image: CUDA runtime libraries only (cudart, cublas, nccl are all in
# the runtime flavor; no nvcc, Nsight, or dev headers).
FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04

# Node.js and npm from the builder (the same binary the addon was built
# against; the tarball layout ships npm alongside node).
COPY --from=builder /usr/local/bin/node /usr/local/bin/npm /usr/local/bin/npx /usr/local/bin/
COPY --from=builder /usr/local/lib/node_modules /usr/local/lib/node_modules

WORKDIR /build/glm.js

# The pruned tree: dist/ (compiled JS), src/ (for tsx entry points),
# build/Release/{glm.node,libglm_ops.so}, node_modules without onnx, and the
# vendored transformers package's src/types.
COPY --from=builder /build/glm.js .

# Sanity-check the stripped tree at build time: the addon and CUDA ops library
# must be present, all shared-library dependencies (cudart/cublas/nccl from
# the runtime base) must resolve, and the tokenizers module graph root,
# compiled JS, and tsx must survive the slimming.
RUN test -f build/Release/glm.node \
    && test -f build/Release/libglm_ops.so \
    && test -f dist/src/main.js \
    && test -d node_modules/tsx \
    && test -d node_modules/@huggingface/jinja \
    && test -d node_modules/@huggingface/tokenizers \
    && test -f node_modules/@huggingface/transformers/package.json \
    && test -f vendor/transformers.js/packages/transformers/src/models/auto/tokenization_qwen_auto.js \
    && ! ldd build/Release/libglm_ops.so | grep -q 'not found' \
    && ! ldd build/Release/glm.node | grep -q 'not found'

CMD ["bash"]