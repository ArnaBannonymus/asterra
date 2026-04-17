from __future__ import annotations

import base64
import csv
import hashlib
import os
import tarfile
import zipfile
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Mapping

import tomllib


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    name: str
    version: str
    summary: str
    readme_path: str
    requires_python: str | None
    license_expression: str | None
    license_files: tuple[str, ...]
    authors: tuple[str, ...]
    classifiers: tuple[str, ...]
    dependencies: tuple[str, ...]
    optional_dependencies: Mapping[str, tuple[str, ...]]
    project_urls: Mapping[str, str]

    @property
    def normalized_name(self) -> str:
        return self.name.replace("-", "_")


def get_requires_for_build_wheel(config_settings: Mapping[str, Any] | None = None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings: Mapping[str, Any] | None = None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: Mapping[str, Any] | None = None
) -> str:
    root = Path.cwd()
    meta = _load_metadata(root)
    dist_info = f"{meta.normalized_name}-{meta.version}.dist-info"
    out_dir = Path(metadata_directory) / dist_info
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "METADATA").write_text(_render_metadata(meta, root), encoding="utf-8")
    (out_dir / "WHEEL").write_text(_render_wheel(), encoding="utf-8")
    return dist_info


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    root = Path.cwd()
    meta = _load_metadata(root)
    wheel_dir = Path(wheel_directory)
    wheel_dir.mkdir(parents=True, exist_ok=True)

    dist_info = f"{meta.normalized_name}-{meta.version}.dist-info"
    wheel_name = f"{meta.normalized_name}-{meta.version}-py3-none-any.whl"
    wheel_path = wheel_dir / wheel_name

    records: list[tuple[str, bytes]] = []

    def write_bytes(zf: zipfile.ZipFile, arcpath: str, data: bytes) -> None:
        zf.writestr(arcpath, data)
        records.append((arcpath, data))

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Package files (src-layout)
        pkg_root = root / "src"
        for src_path, arcpath in _iter_package_files(pkg_root):
            data = src_path.read_bytes()
            write_bytes(zf, arcpath, data)

        # dist-info metadata
        write_bytes(zf, f"{dist_info}/METADATA", _render_metadata(meta, root).encode("utf-8"))
        write_bytes(zf, f"{dist_info}/WHEEL", _render_wheel().encode("utf-8"))

        # License files live under dist-info/licenses/
        for lic in meta.license_files:
            lic_path = root / lic
            if lic_path.exists():
                write_bytes(
                    zf,
                    f"{dist_info}/licenses/{Path(lic).name}",
                    lic_path.read_bytes(),
                )

        # RECORD (written last; record line for RECORD has empty hash/size)
        record_path = f"{dist_info}/RECORD"
        record_bytes = _render_record(records, record_path).encode("utf-8")
        zf.writestr(record_path, record_bytes)

    return wheel_name


def build_sdist(sdist_directory: str, config_settings: Mapping[str, Any] | None = None) -> str:
    root = Path.cwd()
    meta = _load_metadata(root)
    sdist_dir = Path(sdist_directory)
    sdist_dir.mkdir(parents=True, exist_ok=True)

    base = f"{meta.normalized_name}-{meta.version}"
    out_path = sdist_dir / f"{base}.tar.gz"

    with tarfile.open(out_path, "w:gz") as tf:
        # Add PKG-INFO
        pkg_info = _render_metadata(meta, root).encode("utf-8")
        _tar_add_bytes(tf, f"{base}/PKG-INFO", pkg_info)

        for rel_path in _iter_sdist_files(root):
            arcname = f"{base}/{rel_path.as_posix()}"
            tf.add(root / rel_path, arcname=arcname, recursive=False)

    return out_path.name


def _load_metadata(root: Path) -> ProjectMetadata:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})

    name = str(project["name"])
    version = str(project["version"])
    summary = str(project.get("description", ""))
    readme_path = project.get("readme", "README.md")
    requires_python = project.get("requires-python")
    license_expression = project.get("license")
    if license_expression is not None:
        license_expression = str(license_expression)

    license_files = tuple(project.get("license-files", []))
    authors = tuple(a.get("name", "") for a in project.get("authors", []) if a.get("name"))
    classifiers = tuple(project.get("classifiers", []))
    dependencies = tuple(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {}) or {}
    optional_deps = {k: tuple(v) for (k, v) in optional.items()}
    project_urls = project.get("urls", {}) or {}

    return ProjectMetadata(
        name=name,
        version=version,
        summary=summary,
        readme_path=str(readme_path),
        requires_python=str(requires_python) if requires_python is not None else None,
        license_expression=license_expression,
        license_files=license_files,
        authors=authors,
        classifiers=classifiers,
        dependencies=dependencies,
        optional_dependencies=optional_deps,
        project_urls=dict(project_urls),
    )


def _render_metadata(meta: ProjectMetadata, root: Path) -> str:
    lines: list[str] = []
    lines.append("Metadata-Version: 2.4")
    lines.append(f"Name: {meta.name}")
    lines.append(f"Version: {meta.version}")
    if meta.summary:
        lines.append(f"Summary: {meta.summary}")
    if meta.requires_python:
        lines.append(f"Requires-Python: {meta.requires_python}")
    if meta.license_expression:
        lines.append(f"License-Expression: {meta.license_expression}")
    for lic in meta.license_files:
        lines.append(f"License-File: {Path(lic).name}")
    if meta.authors:
        lines.append(f"Author: {', '.join(meta.authors)}")
    for classifier in meta.classifiers:
        lines.append(f"Classifier: {classifier}")
    for dep in meta.dependencies:
        lines.append(f"Requires-Dist: {dep}")
    for extra, reqs in meta.optional_dependencies.items():
        lines.append(f"Provides-Extra: {extra}")
        for req in reqs:
            lines.append(f'Requires-Dist: {req}; extra == "{extra}"')
    for label, url in meta.project_urls.items():
        lines.append(f"Project-URL: {label}, {url}")

    readme_path = root / meta.readme_path
    if readme_path.exists():
        lines.append("Description-Content-Type: text/markdown")
        lines.append("")
        lines.append(readme_path.read_text(encoding="utf-8"))
    else:
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_wheel() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: asterra-build-backend (pure-python)",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _iter_package_files(src_root: Path) -> list[tuple[Path, str]]:
    """Return (src_path, arcpath) entries for wheel inclusion."""

    out: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith((".pyc", ".pyo")):
                continue
            p = Path(dirpath) / fn
            rel = p.relative_to(src_root).as_posix()
            out.append((p, rel))
    out.sort(key=lambda t: t[1])
    return out


def _iter_sdist_files(root: Path) -> list[Path]:
    excluded = {
        ".git",
        ".venv",
        ".pytest_cache",
        "build",
        "dist",
        "build_artifacts",
        "__pycache__",
    }
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        if rel_dir.parts and rel_dir.parts[0] in excluded:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in excluded and d != "__pycache__"]
        for fn in filenames:
            if fn.endswith((".pyc", ".pyo")):
                continue
            p = Path(dirpath) / fn
            if fn in {".DS_Store"}:
                continue
            files.append(p.relative_to(root))
    files.sort(key=lambda p: p.as_posix())
    return files


def _tar_add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tf.addfile(info, BytesIO(data))


def _render_record(records: list[tuple[str, bytes]], record_path: str) -> str:
    sio = StringIO()
    writer = csv.writer(sio, lineterminator="\n")
    for path, data in records:
        digest = hashlib.sha256(data).digest()
        b64 = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        writer.writerow([path, f"sha256={b64}", str(len(data))])
    writer.writerow([record_path, "", ""])
    return sio.getvalue()
