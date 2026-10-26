# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve a branch once and request an image-only Buildkite build."""

import json
import os
import string
import urllib.parse
import urllib.request


def request_build(
    repository: str,
    ref: str,
    pipeline: str,
    organization: str,
    github_token: str,
    buildkite_token: str,
    image_repository: str,
    architecture: str = "amd64",
    queue: str = "",
) -> dict:
    if architecture not in ("amd64", "arm64") or not queue:
        raise ValueError(
            "An amd64/arm64 architecture and matching CPU build queue are required"
        )
    parts = repository.split("/")
    allowed = set(string.ascii_letters + string.digits + "-._")
    if len(parts) != 2 or any(not p or not set(p) <= allowed for p in parts):
        raise ValueError("Invalid GitHub repository")
    if not ref or not pipeline or not organization or not image_repository:
        raise ValueError(
            "Source ref, Buildkite organization/pipeline "
            "and image repository are required"
        )
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/commits/"
        f"{urllib.parse.quote(ref, safe='')}",
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        sha = json.load(response)["sha"]
    if len(sha) != 40 or not set(sha) <= set("0123456789abcdef"):
        raise ValueError("GitHub did not resolve the ref to a full commit SHA")
    payload = {
        "commit": sha,
        "branch": ref,
        "message": f"BYOG image for {repository}@{sha}",
        "env": {
            "BYOG_REQUESTED_COMMIT": sha,
            "BYOG_ARCH": architecture,
            "BYOG_BUILD_QUEUE": queue,
            "BYOG_IMAGE_REPO": image_repository,
            "BYOG_SOURCE_REPOSITORY": repository,
        },
    }
    request = urllib.request.Request(
        "https://api.buildkite.com/v2/organizations/"
        f"{urllib.parse.quote(organization, safe='')}/pipelines/"
        f"{urllib.parse.quote(pipeline, safe='')}/builds",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {buildkite_token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        build = json.load(response)
    return {"source_commit": sha, "build_url": build["web_url"]}


def main() -> None:
    result = request_build(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["BYOG_SOURCE_REF"],
        os.environ["BUILDKITE_BYOG_PIPELINE"],
        os.environ["BUILDKITE_ORGANIZATION"],
        os.environ["GITHUB_TOKEN"],
        os.environ["BUILDKITE_API_TOKEN"],
        os.environ["BYOG_IMAGE_REPO"],
        os.environ["BYOG_ARCH"],
        os.environ.get(f"BYOG_{os.environ['BYOG_ARCH'].upper()}_QUEUE", ""),
    )
    print(json.dumps(result))
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as output:
            output.write(
                f"BYOG source: `{result['source_commit']}`\n\n"
                f"[Image build]({result['build_url']})\n"
            )


if __name__ == "__main__":
    main()
