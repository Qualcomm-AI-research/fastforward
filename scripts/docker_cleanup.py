# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Queries the Docker registry for all tags of a Docker image and deletes outdated ones.

This script is part of a CI optimization strategy that manages Docker build images
to balance build speed with registry storage. It implements automatic cleanup of
Docker images that are older than a configured retention period (default: 1 week).

Workflow:
    1. Connects to the Docker registry and queries the catalog for all images
       matching a prefix
    2. Retrieves all tags for each matching image
    3. Inspects each tag's manifest to determine the image creation timestamp
    4. Deletes images that exceed the retention period, while preserving protected
       tags (e.g., "latest")

The cleanup strategy enables:
    - Reusing recent build images across CI runs (reducing build time from ~20min to ~1min)
    - Automatic cleanup of stale images to prevent registry bloat
    - Hash-based tagging that allows cache hits when build dependencies are unchanged

Usage:
    This script is typically invoked as part of a scheduled CI job:

    $ python docker_cleanup.py [--dry-run] [--only-one]

Environment Variables:
    DOCKER_REGISTRY: Docker registry hostname
    DOCKER_LOGIN: Docker registry username
    DOCKER_CREDENTIALS: Docker registry password or access token

Security:
    - Requires proper authentication via environment variables
    - Protected tags cannot be deleted
"""

import argparse
import base64
import datetime
import json
import os
import re
import sys
import typing
import urllib.request

_PROTECTED_TAGS = ["latest"]
_OUTDATED_AFTER = datetime.timedelta(weeks=1)
_DOCKER_IMAGE_PREFIX_ENV = "DOCKER_IMAGE"
_MAX_AFFECTED_IMAGES = 10


def main() -> None:
    """Serves as entrypoint."""
    args = _parse_args()
    session = DockerImageQuery()
    session.delete_outdated_tags(dry_run=args.dry_run, only_one=args.only_one)


def _parse_args() -> argparse.Namespace:
    """Parses the CLI arguments."""
    parser = argparse.ArgumentParser(description="Docker cleanup script configuration.")

    # Add boolean flags
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="Run in dry-run mode (default: False)"
    )

    parser.add_argument(
        "--only-one",
        action="store_true",
        default=False,
        help="Abort after one deletion (default: False)",
    )

    return parser.parse_args()


def _get_docker_auth() -> tuple[str, str]:
    """Reads Docker authentication credentials from environment variables.

    Returns:
        A tuple of (username, password) for Docker registry authentication.
    """
    user = os.environ["DOCKER_LOGIN"]
    password = os.environ["DOCKER_CREDENTIALS"]
    return user, password


def _get_image_prefix() -> str:
    image_prefix = os.getenv(_DOCKER_IMAGE_PREFIX_ENV)
    if not image_prefix:
        msg = f"Expected `{_DOCKER_IMAGE_PREFIX_ENV}` to be set."
        raise EnvironmentError(msg)
    return image_prefix


class Response:
    """Encapsulates an HTTP response allowing multiple reads of the content."""

    def __init__(self, content: bytes, headers: dict[str, str]) -> None:
        self.content = content
        self._headers = headers

    def content_as_json(self) -> typing.Any:
        """Parse the response content as JSON."""
        return json.loads(self.content.decode(encoding="utf-8"))

    @property
    def headers(self) -> dict[str, str]:
        """Expose response headers."""
        return dict(self._headers)


class DockerImageQuery:
    """Handles the cleanup of Docker images."""

    def __init__(self) -> None:
        self.outdated_after = _OUTDATED_AFTER

        self.docker_registry = os.environ["DOCKER_REGISTRY"]
        self.docker_hub_api = f"https://{self.docker_registry}/v2"
        self.user, self.pw = _get_docker_auth()
        self.image_name_prefix = _get_image_prefix()

    def _query_image_names(self) -> tuple[str, ...]:
        """Queries the Docker registry catalog for all images we built.

        Returns:
            A tuple of image names that match our prefix.
        """
        "_catalog"
        res = self._make_request(url=self.docker_hub_api, endpoint="/_catalog")
        repos = res.content_as_json()["repositories"]
        ff_images = tuple(repo for repo in repos if repo.startswith(self.image_name_prefix))
        if len(ff_images) > _MAX_AFFECTED_IMAGES:
            images = "\n".join(ff_images)
            print(f"Affected images:\n{images}")
            msg = f"Found more than {_MAX_AFFECTED_IMAGES} for deletion, aborting."
            raise RuntimeError(msg)
        return ff_images

    def _get_image_api(self, image_name: str) -> str:
        return f"https://{self.docker_registry}/v2/{image_name}"

    def _make_request(
        self,
        *,
        endpoint: str,
        image_name: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        method: str = "GET",
        dry_run: bool = False,
    ) -> Response:
        """Makes an authenticated HTTP request and returns content and headers."""
        if bool(image_name) + bool(url) != 1:
            msg = f"Expected either an image name or a full URL, got {image_name=}, {url=}"
            raise ValueError(msg)
        if image_name is not None:
            full_url = self._get_image_api(image_name=image_name) + endpoint
        else:
            assert url is not None
            full_url = url + endpoint
        req_headers = headers.copy() if headers else {}

        auth_str = f"{self.user}:{self.pw}"
        auth_bytes = base64.b64encode(auth_str.encode()).decode("ascii")
        req_headers["Authorization"] = f"Basic {auth_bytes}"

        if dry_run and method != "GET":
            print(f"DRY-RUN: Would execute {method} request against {full_url}")
            return Response(content=b"", headers={})
        else:
            req = urllib.request.Request(full_url, headers=req_headers, method=method)
            with urllib.request.urlopen(req) as res:
                return Response(res.read(), dict(res.headers))

    def delete_outdated_tags(self, dry_run: bool, only_one: bool) -> None:
        """Deletes tags corresponding to outdated Docker images.

        Args:
            dry_run: Only show what would be deleted, but skip deletion.
            only_one: Stop remaining deletions after the first.
        """
        deleted_refs = []
        for image_name in self._query_image_names():
            tags = self._get_tags_for_image(image_name=image_name)
            print(
                f"Considering which tags to delete for image `{image_name}`.\nCandidate tags: "
                + ", ".join(tags)
            )
            for tag in tags:
                if tag in _PROTECTED_TAGS:
                    print(f"Skipping protected tag `{tag}`.")
                    continue
                print(f"Inspecting tag {tag}.")
                manifest = self._get_manifest_for_tag(tag=tag, image_name=image_name)
                if self._is_manifest_outdated(manifest=manifest, image_name=image_name):
                    image_ref = f"{image_name}:{tag}"
                    print(f"Deleting Docker ref `{image_ref}` ...")
                    self._delete_tag(manifest, image_name=image_name, dry_run=dry_run)
                    deleted_refs.append(image_ref)
                    if only_one:
                        print("Stopping after first deletion.")
                        return
                else:
                    print(f"Skipping, image is younger than {self.outdated_after}.")
        if deleted_refs:
            dry_run_prefix = "DRY-RUN:" if dry_run else ""
            print(f"{dry_run_prefix} List of deleted tags:\n" + "\n".join(deleted_refs))
        else:
            print("No images were deleted.")

    def _get_tags_for_image(self, image_name: str) -> list[str]:
        """Gets the list of tags of a given Docker image."""
        print(f"Querying Docker registry for list of all tags for image `{image_name}`.")
        resp = self._make_request(image_name=image_name, endpoint="/tags/list")
        tags = resp.content_as_json()["tags"]
        assert _is_list_str(tags)
        return tags

    def _get_manifest_for_tag(self, tag: str, image_name: str) -> Response:
        """Gets a manifest for a given tag."""
        headers = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
        return self._make_request(
            image_name=image_name, endpoint=f"/manifests/{tag}", headers=headers
        )

    def _is_manifest_outdated(self, manifest: Response, image_name: str) -> bool:
        """Determines if a Docker manifest represents an outdated image.

        Args:
            manifest: The manifest response containing image metadata.
            image_name: The name of the Docker image.

        Returns:
            True if the image creation date exceeds the retention period, False otherwise.
        """
        digest = manifest.content_as_json()["config"]["digest"]

        resp = self._make_request(image_name=image_name, endpoint=f"/blobs/{digest}")
        created_at_str = resp.content_as_json()["created"]

        created_at = _from_isoformat(created_at_str)
        now = datetime.datetime.now(tz=created_at.tzinfo)

        delta = abs(now - created_at)
        outdated = delta > self.outdated_after
        if outdated:
            print(f"Outdated by {delta.days} days.")
        return outdated

    def _delete_tag(self, manifest: Response, image_name: str, dry_run: bool) -> None:
        """Deletes a Docker image tag based on its manifest.

        Args:
            manifest: The manifest response containing the content digest.
            image_name: The name of the Docker image.
            dry_run: If True, only print what would be deleted without performing deletion.
        """
        content_digest = manifest.headers["Docker-Content-Digest"]
        self._make_request(
            image_name=image_name,
            endpoint=f"/manifests/{content_digest}",
            method="DELETE",
            dry_run=dry_run,
        )


def _from_isoformat(ts: str) -> datetime.datetime:
    """Parse 8601 timestamp with nanosecond precision (Python 3.10 compatibility helper)."""
    if sys.version_info <= (3, 10):
        # Truncate fractional seconds to 6 digits and make TZ format compatible.
        modified = re.sub(r"\.(\d{6})\d+", r".\1", ts).replace("Z", "+00:00")
    else:
        modified = ts
    return datetime.datetime.fromisoformat(modified)


def _is_list_str(val: typing.Any) -> typing.TypeGuard[list[str]]:
    """Type guards val through an explicit runtime check."""
    return isinstance(val, list) and all(isinstance(x, str) for x in val)


if __name__ == "__main__":
    main()
