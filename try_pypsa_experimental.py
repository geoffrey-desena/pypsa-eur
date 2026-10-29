"""Run from the pypsa-eur project root in Spyder (F5).

Uses only Python's standard library in Spyder. Pixi runs Snakemake in the
project environment, so Spyder need not use that environment itself.

Reads the experimental example and original downloader from LOCAL git HEAD.
If the downloader was edited, saves a backup before restoring that one file.
Keeps downloaded data; does not update git, packages, or other source files.
Uses the standard Overpass workflow, with no fallback to another data source.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess


# -------------------------- EDIT SETTINGS HERE --------------------------
PROJECT_ROOT = None  # None = folder containing this script; or an absolute path
COUNTRIES = ["HU"]
PROJECT_NAME = "GeoffreyDeSena-GridResearch/1.0"
CONTACT_EMAIL = "geoffrey.desena@angstrom.uu.se"
PROJECT_WEBSITE = ""
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAX_TRIES = 5
QUERY_TIMEOUT_SECONDS = 600
CORES = 1
RUN_NAME = "sweden-experimental"
DRY_RUN_ONLY = False  # True previews jobs without executing the build
RESTORE_ORIGINAL_DOWNLOADER = True  # Backs up edits before restoring git HEAD
PIXI_EXECUTABLE = None  # Set an absolute path if automatic discovery fails
# -----------------------------------------------------------------------


def git_bytes(root, *args):
    """Read local git metadata; never fetch or modify git history."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot read local git checkout: {error}") from error


def find_pixi():
    candidates = [
        PIXI_EXECUTABLE,
        shutil.which("pixi"),
        Path.home() / ".pixi/bin/pixi",
        Path("/opt/homebrew/bin/pixi"),
        Path("/usr/local/bin/pixi"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise RuntimeError(
        "Pixi was not found. Set PIXI_EXECUTABLE near the top of this script "
        "to the path printed by 'which pixi' in Terminal."
    )


def restore_downloader(root, session):
    relative = "scripts/retrieve_osm_data.py"
    original = git_bytes(root, "show", f"HEAD:{relative}")
    destination = root / relative
    if destination.exists() and destination.read_bytes() == original:
        print("Downloader matches local git HEAD.", flush=True)
        return
    if not RESTORE_ORIGINAL_DOWNLOADER:
        raise RuntimeError(
            "Downloader differs from local git HEAD. For this standard-workflow "
            "test, set RESTORE_ORIGINAL_DOWNLOADER=True to back up and restore it."
        )
    if destination.exists():
        backup = session / "retrieve_osm_data.before.py"
        shutil.copy2(destination, backup)
        print(f"Previous downloader backed up to: {backup}", flush=True)
    destination.write_bytes(original)
    print("Restored ONLY scripts/retrieve_osm_data.py from local git HEAD.", flush=True)


def run_live(command, root, log_path):
    """Stream the child output to Spyder and a log; propagate interruptions."""
    print("\nCommand: " + shlex.join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    # Spyder may itself run in a different Python environment.
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        command, cwd=root, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        bufsize=1, start_new_session=(os.name == "posix"),
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(shlex.join(command) + "\n")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        return process.wait()
    finally:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        process.stdout.close()


def main():
    root = (Path(PROJECT_ROOT).expanduser() if PROJECT_ROOT else
            Path(__file__).resolve().parent).resolve()
    for filename in ("Snakefile", "pixi.toml", "config/config.default.yaml"):
        if not (root / filename).is_file():
            raise RuntimeError(
                f"Missing {root / filename}. Save this script in the pypsa-eur "
                "root, or set PROJECT_ROOT to that folder."
            )
    if not COUNTRIES or any(not isinstance(c, str) or len(c) != 2 for c in COUNTRIES):
        raise ValueError("COUNTRIES must contain two-letter country codes")
    if not RUN_NAME or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in RUN_NAME):
        raise ValueError("RUN_NAME must contain only letters, numbers, '_' or '-'")
    if MAX_TRIES < 1 or QUERY_TIMEOUT_SECONDS < 1 or CORES < 1:
        raise ValueError("Retry count, query timeout, and cores must be positive")
    pixi = find_pixi()
    example = git_bytes(
        root, "show", "HEAD:config/examples/config.distribution-grid-experimental.yaml"
    )
    commit = git_bytes(root, "rev-parse", "HEAD").decode().strip()
    session = root / "logs" / "experimental_launcher" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session.mkdir(parents=True)
    print(f"Project: {root}\nLocal git HEAD: {commit}\nLogs: {session}", flush=True)
    restore_downloader(root, session)

    # Separate config files use Snakemake's documented merge/override mechanism.
    # JSON literals are valid YAML, avoiding any extra dependency in Spyder.
    example_path = session / "experimental_from_git.yaml"
    example_path.write_bytes(example)
    overrides = {
        "countries": COUNTRIES,
        "run": {
            "name": RUN_NAME, "prefix": "",
            "scenarios": {"enable": False},
            "shared_resources": {"policy": False, "exclude": []},
        },
        "overpass_api": {
            "url": OVERPASS_URL, "max_tries": MAX_TRIES,
            "timeout": QUERY_TIMEOUT_SECONDS,
            "user_agent": {"project_name": PROJECT_NAME,
                           "email": CONTACT_EMAIL, "website": PROJECT_WEBSITE},
        },
    }
    override_path = session / "local_overrides.yaml"
    override_path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    if (root / "config/config.yaml").exists():
        print("Note: config/config.yaml exists and may supply additional settings. "
              "The two explicit config files override matching keys.", flush=True)
    base = [pixi, "run", "snakemake", "build_osm_network", "--configfile",
            str(example_path), str(override_path), "--cores", str(CORES),
            "--rerun-incomplete"]
    print("Running the dry run first. Network metadata checks can still occur.", flush=True)
    if run_live(base + ["--dry-run"], root, session / "dry_run.log"):
        print(f"Dry run failed. See {session / 'dry_run.log'}", flush=True)
        return
    if DRY_RUN_ONLY:
        print("Dry run completed; DRY_RUN_ONLY=True, so no build was started.", flush=True)
        return
    print("\nStarting the build. Server requests may be silent for several minutes.\n"
          "There are no extra launcher retries or automatic server switches.", flush=True)
    status = run_live(base, root, session / "build.log")
    if status:
        print(f"\nBuild FAILED (exit {status}). Full log: {session / 'build.log'}", flush=True)
        print("Completed supporting downloads have been kept.", flush=True)
        return
    output = root / "resources" / RUN_NAME / "osm" / "build"
    print(f"\nBuild completed. Network outputs: {output}", flush=True)
    for name in ("buses.csv", "lines.csv", "transformers.csv", "links.csv"):
        path = output / name
        print(f"  {name}: {'present' if path.is_file() else 'not found at expected path'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Check the session logs before restarting.", flush=True)
