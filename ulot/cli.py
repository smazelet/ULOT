"""Console entry points for the paper reproduction scripts."""

from __future__ import annotations

import runpy
import shutil
import sysconfig
from importlib.util import find_spec
from pathlib import Path


_ARTIFACT_DIRS = ("parameter_files", "results")


def _has_artifacts(path: Path) -> bool:
    return all((path / dirname).exists() for dirname in _ARTIFACT_DIRS)


def _candidate_artifact_roots(module_name: str) -> list[Path]:
    roots = []
    spec = find_spec(module_name)
    if spec is not None and spec.origin is not None:
        roots.append(Path(spec.origin).resolve().parent)

    data_path = sysconfig.get_path("data")
    if data_path is not None:
        roots.append(Path(data_path).resolve() / "share" / "ulot")

    return roots


def _prepare_artifacts(module_name: str) -> None:
    cwd = Path.cwd()
    if _has_artifacts(cwd):
        return

    for root in _candidate_artifact_roots(module_name):
        if not _has_artifacts(root):
            continue
        for dirname in _ARTIFACT_DIRS:
            source = root / dirname
            target = cwd / dirname
            if not target.exists():
                shutil.copytree(source, target)
        return


def _run_script_module(module_name: str) -> None:
    _prepare_artifacts(module_name)
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)


def accuracy_surfaces() -> None:
    _run_script_module("accuracy_surfaces")


def effect_rho_alpha() -> None:
    _run_script_module("effect_rho_alpha")


def functional_minimization() -> None:
    _run_script_module("functional_minimization")


def label_propagation() -> None:
    _run_script_module("label_propagation")


def similarity_mass() -> None:
    _run_script_module("similarity_mass")


def solve() -> None:
    _run_script_module("run_ot_solver")


def test() -> None:
    _run_script_module("run_test")


def train() -> None:
    _run_script_module("run_train")


def visus() -> None:
    _run_script_module("visus")
