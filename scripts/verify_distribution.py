"""Verify package identity, then execute tests using installed modules outside source."""

from __future__ import annotations
import argparse
from email.parser import BytesParser
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    wheel, sdist = args.wheel.resolve(strict=True), args.sdist.resolve(strict=True)
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(
        PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg"
    )
    with tempfile.TemporaryDirectory(prefix="healing-installed-") as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        with tarfile.open(sdist) as archive:
            members = archive.getmembers()
            if (
                len(members) > 10000
                or sum(m.size for m in members) > 64 * 1024 * 1024
                or len({m.name for m in members}) != len(members)
            ):
                raise ValueError("invalid or excessive source archive")
            archive.extractall(source, filter="data")
        roots = list(source.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("expected one source root")
        project = roots[0]
        metadata = tomllib.loads((project / "pyproject.toml").read_text())["project"]
        name, version = metadata["name"], metadata["version"]

        def identity(raw):
            parsed = BytesParser().parsebytes(raw)
            if parsed.get_all("Name") != [name] or parsed.get_all("Version") != [
                version
            ]:
                raise ValueError("invalid distribution identity")

        identity((project / "PKG-INFO").read_bytes())
        reference = {}
        for directory, prefix in (
            (
                project / "src/self_healing_embodied_agents",
                "self_healing_embodied_agents",
            ),
            (project / "scripts", "scripts"),
        ):
            reference.update(
                {
                    prefix + "/" + p.relative_to(directory).as_posix(): p.read_bytes()
                    for p in directory.rglob("*.py")
                }
            )
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate wheel members")
            meta = [n for n in names if n.endswith(".dist-info/METADATA")]
            if len(meta) != 1:
                raise ValueError("ambiguous wheel metadata")
            identity(archive.read(meta[0]))
            actual = {
                n: archive.read(n)
                for n in names
                if n.endswith(".py")
                and n.split("/")[0] in {"self_healing_embodied_agents", "scripts"}
            }
            if actual != reference:
                raise ValueError("wheel/source module bytes differ")
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        work = root / "work"
        work.mkdir()
        # Historical tests intentionally read checked data by relative path.
        shutil.copytree(project / "results", work / "results")
        shutil.copytree(project / "configs", work / "configs")
        shutil.copytree(project / "artifacts", work / "artifacts")

        def run(arguments):
            subprocess.run(
                [str(python), "-I", *arguments],
                cwd=work,
                env=env,
                check=True,
                timeout=300,
            )

        if sys.platform == "linux":
            run(
                [
                    "-m",
                    "pip",
                    "install",
                    "torch>=2.6,<3",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ]
            )
        run(["-m", "pip", "install", str(wheel), "pytest>=8,<10"])
        run(["-m", "pip", "check"])
        check_script = """import pathlib,sys,importlib.metadata as md
import self_healing_embodied_agents as package, scripts
assert package.__version__==md.version(sys.argv[1])==sys.argv[2]
project=pathlib.Path(sys.argv[3])
for module,source in [(package,project/'src/self_healing_embodied_agents'),(scripts,project/'scripts')]:
 installed=pathlib.Path(module.__file__).resolve().parent
 assert pathlib.Path(sys.prefix).resolve() in installed.parents
 assert {p.relative_to(source).as_posix():p.read_bytes() for p in source.rglob('*.py')}=={p.relative_to(installed).as_posix():p.read_bytes() for p in installed.rglob('*.py')}
 print('Installed modules verified',installed)
"""
        run(["-c", check_script, name, version, str(project)])
        # importlib mode and a separate cwd prevent checkout scripts from masking the wheel.
        run(["-m", "pytest", "--import-mode=importlib", "-q", str(project / "tests")])
        run(["-m", "scripts.showcase", "--model", str(project/"artifacts/world_model.pt"), "--out", str(root/"showcase")])
        run(["-m", "scripts.verify_comparison", str(project/"results/comparison-v1/evaluation"), "--replay-actions"])
        report = {
            "status": "PASS",
            "distribution": name,
            "version": version,
            "runtime_and_experiment_modules_matched": len(reference),
            "installation": "fresh non-editable wheel, runtime and experiment imports outside source tree",
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "sdist_sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
