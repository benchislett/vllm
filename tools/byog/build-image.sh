#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
: "${BYOG_REQUESTED_COMMIT:?request must pin a commit}"
: "${BYOG_IMAGE_REPO:?configure a dedicated BYOG ECR repository}"
: "${BYOG_SOURCE_REPOSITORY:?GitHub repository identity is required}"
: "${BUILDKITE_BUILD_NUMBER:?run this on a Buildkite image builder}"
[[ "$(git rev-parse HEAD)" == "$BYOG_REQUESTED_COMMIT" ]] || { echo 'Checkout differs from requested SHA' >&2; exit 1; }
case "$BYOG_IMAGE_REPO" in
  vllm-ci-*|vllm-release-*) echo 'BYOG requires a separate image repository' >&2; exit 1 ;;
esac
registry="${BYOG_REGISTRY:-public.ecr.aws/q9t5s3a7}"
base_repo=vllm-ci-test-repo
base_image="$registry/$base_repo:$BYOG_REQUESTED_COMMIT"
export TORCH_NIGHTLY=0
architecture="${BYOG_ARCH:-amd64}"
export BUILDKITE_PULL_REQUEST="${BUILDKITE_PULL_REQUEST:-false}"
export CI_HCL_URL="${CI_HCL_URL:-https://raw.githubusercontent.com/vllm-project/ci-infra/e28b4ff355e758df18052f27a170c8af6c6676e6/docker/ci.hcl}"
case "$architecture" in
  amd64)
    bash .buildkite/image_build/image_build.sh "$registry" "$base_repo" "$BYOG_REQUESTED_COMMIT" "$BUILDKITE_BRANCH" "$base_image"
    ;;
  arm64)
    base_image="$base_image-arm64"
    bash .buildkite/image_build/image_build_arm64.sh "$registry" "$base_repo" "$BYOG_REQUESTED_COMMIT"
    ;;
  *) echo 'Unsupported architecture' >&2; exit 1 ;;
esac
# Buildx resolves the manifest digest without downloading filesystem layers.
base_digest="$(docker buildx imagetools inspect "$base_image" --format '{{.Manifest.Digest}}')"
[[ "$base_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'Invalid base digest' >&2; exit 1; }
# Verify this exact digest resolves before using it as a base.
docker buildx imagetools inspect "$registry/$base_repo@$base_digest" >/dev/null
image_tag="$registry/$BYOG_IMAGE_REPO:byog-$BYOG_REQUESTED_COMMIT-$architecture-$BUILDKITE_BUILD_NUMBER"
build_context="$(mktemp -d)"
trap 'rm -rf "$build_context"' EXIT
mkdir -p "$build_context/tools" "$build_context/data"
cp -r tools/byog tools/recipes "$build_context/tools/"
cp docker/Dockerfile.byog "$build_context/Dockerfile"
python3 tools/byog/prepare_image.py "$build_context" "$base_digest"
docker buildx build --platform "linux/$architecture" --file "$build_context/Dockerfile" \
  --build-arg "BASE_IMAGE=$registry/$base_repo@$base_digest" \
  --tag "$image_tag" --metadata-file "$build_context/metadata.json" \
  --provenance=mode=max --push "$build_context"
python3 - "$build_context/metadata.json" "$image_tag" <<'PY'
import json,sys
from pathlib import Path
metadata=json.loads(Path(sys.argv[1]).read_text())
result={'image':sys.argv[2], 'digest':metadata['containerimage.digest'], 'retention':'manual'}
Path('byog-image.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
bash .buildkite/scripts/annotate-image-build.sh "$image_tag"
