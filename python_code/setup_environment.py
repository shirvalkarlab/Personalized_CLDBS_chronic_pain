"""
setup_environment.py - one-step Python environment setup for this deposit.

Run this ONCE before running any of the Python figure or analysis scripts. It
builds a self-contained environment with the exact package versions the
figures were made with. You can start it with any Python 3 you have.

WHAT IT DOES
  1. Finds a supported Python (3.10 or 3.11; the pinned SciPy does not install
     on 3.12 or later). It uses, in order: the Python running this script; a
     python3.11 or python3.10 already on this computer (PATH, the Windows "py"
     launcher, pyenv, Homebrew); or, if there is none, it downloads Python 3.11
     with uv (https://docs.astral.sh/uv/), which it fetches with pip into a
     temporary folder. The downloaded Python goes into uv's own folder and does
     not replace or change any Python you already have.
  2. Creates a virtual environment in this folder (python_code/.venv) with that
     Python. An existing .venv made with an unsupported Python is replaced.
  3. Installs the exact package versions listed in requirements.txt into it.
  4. Checks that every installed version matches requirements.txt, and prints
     the command that activates the environment.

USAGE
  # from the python_code/ folder, with any Python 3:
  python3 setup_environment.py          (Windows: py setup_environment.py)

  # then activate the environment it created:
  #   macOS / Linux:   source .venv/bin/activate
  #   Windows:         .venv\\Scripts\\activate

  # to install into an environment you already manage instead of making a new
  # one (for example an activated conda environment with Python 3.10 or 3.11):
  python setup_environment.py --no-venv

An internet connection is needed to download packages (and Python, if no
supported version is installed).
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile


SUPPORTED_VERSIONS = ((3, 10), (3, 11))
DOWNLOAD_VERSION = "3.11"                       # what uv installs when none is found

# Folder holding this script (python_code/), and the files beside it.
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(SCRIPT_DIRECTORY, "requirements.txt")
VENV_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, ".venv")


# ---------------------------------------------------------------------------
# Finding a supported Python
# ---------------------------------------------------------------------------

def python_version(executable):
    """Return (major, minor) of a Python executable, or None if it does not run."""
    try:
        output = subprocess.run(
            [executable, "-c", "import sys; print('%d %d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if output.returncode != 0:
        return None
    try:
        major, minor = output.stdout.split()
        return int(major), int(minor)
    except ValueError:
        return None


def is_supported(version):
    return version in SUPPORTED_VERSIONS


def candidate_pythons():
    """Python executables to try, most preferred first (the running one first)."""
    candidates = [sys.executable]
    for name in ("python3.11", "python3.10"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    if os.name == "nt":
        # The Windows "py" launcher knows every python.org install.
        for version in ("3.11", "3.10"):
            try:
                output = subprocess.run(["py", "-" + version, "-c", "import sys; print(sys.executable)"],
                                        capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                continue
            if output.returncode == 0 and output.stdout.strip():
                candidates.append(output.stdout.strip())
    home = os.path.expanduser("~")
    patterns = [os.path.join(home, ".pyenv", "versions", "3.1[01].*", "bin", "python"),
                "/opt/homebrew/opt/python@3.1[01]/bin/python3.1[01]",
                "/usr/local/opt/python@3.1[01]/bin/python3.1[01]",
                os.path.join(home, ".local", "share", "uv", "python", "cpython-3.1[01].*", "bin", "python3")]
    for pattern in patterns:
        candidates.extend(sorted(glob.glob(pattern), reverse=True))
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def find_supported_python():
    """Return the first supported Python executable on this computer, or None."""
    for candidate in candidate_pythons():
        if is_supported(python_version(candidate)):
            return candidate
    return None


def download_python_with_uv():
    """Install Python 3.11 with uv and return its executable, or None on failure."""
    print("No Python 3.10 or 3.11 found. Downloading Python %s with uv ..." % DOWNLOAD_VERSION)
    uv = shutil.which("uv")
    temporary = None
    if uv is None:
        temporary = tempfile.mkdtemp(prefix="uv_bootstrap_")
        print("  Fetching uv with pip into a temporary folder ...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                                 "--disable-pip-version-check", "--target", temporary, "uv"])
        if result.returncode != 0:
            print("  Could not install uv with pip.")
            return None
        found = (glob.glob(os.path.join(temporary, "bin", "uv"))
                 + glob.glob(os.path.join(temporary, "bin", "uv.exe"))
                 + glob.glob(os.path.join(temporary, "**", "uv.exe"), recursive=True))
        if not found:
            print("  uv was installed but its program was not found.")
            return None
        uv = found[0]
    try:
        if subprocess.run([uv, "python", "install", DOWNLOAD_VERSION]).returncode != 0:
            print("  uv could not download Python %s." % DOWNLOAD_VERSION)
            return None
        output = subprocess.run([uv, "python", "find", DOWNLOAD_VERSION],
                                capture_output=True, text=True)
        executable = output.stdout.strip()
    finally:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)
    if output.returncode == 0 and is_supported(python_version(executable)):
        print("  Installed: %s" % executable)
        return executable
    print("  The downloaded Python could not be found or run.")
    return None


def fail_without_python():
    print("\nERROR: no Python 3.10 or 3.11 is available, and none could be downloaded.")
    print("Install one, then run this script again with it. Any of these works:")
    print("  - python.org:  https://www.python.org/downloads/release/python-3119/")
    print("  - conda:       conda create -n natmed python=3.11 && conda activate natmed")
    print("  - uv:          uv python install 3.11")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Virtual environment and packages
# ---------------------------------------------------------------------------

def python_inside_environment():
    """Return the path to the python executable inside the virtual environment."""
    if os.name == "nt":
        return os.path.join(VENV_DIRECTORY, "Scripts", "python.exe")
    return os.path.join(VENV_DIRECTORY, "bin", "python")


def create_virtual_environment(base_python):
    """Create .venv with base_python, replacing one made with an unsupported Python."""
    if os.path.isdir(VENV_DIRECTORY):
        existing = python_version(python_inside_environment())
        if is_supported(existing):
            print("Using the existing virtual environment: %s (Python %d.%d)"
                  % ((VENV_DIRECTORY,) + existing))
            return
        print("The existing %s uses %s, which is not supported; replacing it."
              % (VENV_DIRECTORY, "Python %d.%d" % existing if existing else "a Python that does not run"))
        shutil.rmtree(VENV_DIRECTORY)
    print("Creating a virtual environment in %s with %s ..." % (VENV_DIRECTORY, base_python))
    subprocess.check_call([base_python, "-m", "venv", VENV_DIRECTORY])


def read_pins(requirements_file):
    """Return {package: version} for every 'package==version' line."""
    pins = {}
    with open(requirements_file) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if "==" in line:
                name, version = line.split("==", 1)
                pins[name.strip()] = version.strip()
    return pins


def install_requirements(target_python):
    """Upgrade pip, then install the pinned requirements with the given python."""
    if not os.path.isfile(REQUIREMENTS_FILE):
        print("ERROR: requirements.txt not found next to this script.")
        sys.exit(1)
    print("Upgrading pip ...")
    subprocess.check_call([target_python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    print("Installing pinned requirements from requirements.txt (a few minutes) ...")
    subprocess.check_call([target_python, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])


def check_installed_versions(target_python):
    """Exit with an error unless every pinned package is installed at its pinned version."""
    pins = read_pins(REQUIREMENTS_FILE)
    script = ("import json, sys\n"
              "from importlib.metadata import version, PackageNotFoundError\n"
              "out = {}\n"
              "for name in json.loads(sys.argv[1]):\n"
              "    try: out[name] = version(name)\n"
              "    except PackageNotFoundError: out[name] = None\n"
              "print(json.dumps(out))\n")
    import json
    output = subprocess.run([target_python, "-c", script, json.dumps(list(pins))],
                            capture_output=True, text=True, check=True)
    installed = json.loads(output.stdout)
    wrong = {name: (pins[name], installed[name]) for name in pins if installed[name] != pins[name]}
    if wrong:
        print("\nERROR: these packages do not match requirements.txt:")
        for name, (wanted, got) in sorted(wrong.items()):
            print("  %-16s wanted %s, installed %s" % (name, wanted, got))
        sys.exit(1)
    print("Checked: all %d pinned packages are installed at their pinned versions." % len(pins))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Set up the Python environment for the Shirvalkar 2026 deposit.")
    parser.add_argument(
        "--no-venv", action="store_true",
        help="Install into the current Python instead of creating a new "
             "virtual environment (the current Python must be 3.10 or 3.11).")
    arguments = parser.parse_args()

    running = python_version(sys.executable)
    print("Running Python: %d.%d (%s)" % (running + (sys.executable,)))

    if arguments.no_venv:
        if not is_supported(running):
            print("ERROR: --no-venv installs into this Python, but it is %d.%d; "
                  "the pinned packages need 3.10 or 3.11. Run without --no-venv "
                  "to let this script find or download a supported Python." % running)
            sys.exit(1)
        install_requirements(sys.executable)
        check_installed_versions(sys.executable)
        print("\nDone. You can now run the figure scripts with this Python.")
        return

    venv_version = python_version(python_inside_environment()) if os.path.isdir(VENV_DIRECTORY) else None
    if is_supported(venv_version):
        base_python = python_inside_environment()        # reuse the existing environment
    else:
        base_python = find_supported_python() or download_python_with_uv()
        if base_python is None:
            fail_without_python()
        print("Using Python %d.%d: %s" % (python_version(base_python) + (base_python,)))

    create_virtual_environment(base_python)
    install_requirements(python_inside_environment())
    check_installed_versions(python_inside_environment())

    print("\nDone. Activate the environment before running the figure scripts:")
    if os.name == "nt":
        print("    %s\\Scripts\\activate" % VENV_DIRECTORY)
    else:
        print("    source %s/bin/activate" % VENV_DIRECTORY)
    print("Then, for example:")
    print("    python plot_figureS6_battery_efficiency.py")


if __name__ == "__main__":
    main()
