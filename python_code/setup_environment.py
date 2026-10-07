"""
setup_environment.py - one-step Python environment setup for this deposit.

Run this ONCE before running any of the Python figure or analysis scripts. It
makes sure you have a working, correctly-versioned Python environment so the
figures reproduce exactly.

WHAT IT DOES
  1. Checks that your Python is a supported version (3.10 or 3.11 recommended;
     the pinned package versions were tested on 3.10).
  2. Creates a self-contained virtual environment in this folder
     (python_code/.venv) unless one already exists, so the deposit's packages
     never disturb your system Python.
  3. Installs the exact package versions listed in requirements.txt into that
     virtual environment.
  4. Prints the one command you use to activate the environment afterward.

USAGE
  # from the python_code/ folder:
  python setup_environment.py

  # then activate the environment it created:
  #   macOS / Linux:   source .venv/bin/activate
  #   Windows:         .venv\\Scripts\\activate

  # to install into an environment you already manage instead of making a new
  # one (for example a conda environment you have activated), pass --no-venv:
  python setup_environment.py --no-venv
"""

import os
import sys
import subprocess
import argparse


# Folder holding this script (python_code/), and the requirements file beside it.
script_directory = os.path.dirname(os.path.abspath(__file__))
requirements_file = os.path.join(script_directory, "requirements.txt")
virtual_environment_directory = os.path.join(script_directory, ".venv")


def check_python_version():
    """Warn if the running Python is outside the tested range (3.10-3.11)."""
    major, minor = sys.version_info.major, sys.version_info.minor
    print("Python interpreter: %d.%d (%s)" % (major, minor, sys.executable))
    if major != 3 or minor not in (10, 11):
        print("  WARNING: the pinned package versions were tested on Python "
              "3.10 (3.11 also works). Another version may fail to install "
              "some pinned packages. Continuing anyway.")
    else:
        print("  Supported Python version.")


def python_inside_environment():
    """Return the path to the python executable inside the virtual environment."""
    if os.name == "nt":
        return os.path.join(virtual_environment_directory, "Scripts", "python.exe")
    return os.path.join(virtual_environment_directory, "bin", "python")


def create_virtual_environment():
    """Create the virtual environment if it does not already exist."""
    if os.path.isdir(virtual_environment_directory):
        print("Virtual environment already exists: %s" % virtual_environment_directory)
        return
    print("Creating virtual environment in %s ..." % virtual_environment_directory)
    subprocess.check_call([sys.executable, "-m", "venv", virtual_environment_directory])
    print("  Created.")


def install_requirements(target_python):
    """Upgrade pip, then install the pinned requirements with the given python."""
    if not os.path.isfile(requirements_file):
        print("ERROR: requirements.txt not found next to this script.")
        sys.exit(1)
    print("Upgrading pip ...")
    subprocess.check_call([target_python, "-m", "pip", "install", "--upgrade", "pip"])
    print("Installing pinned requirements from requirements.txt ...")
    subprocess.check_call([target_python, "-m", "pip", "install",
                           "-r", requirements_file])
    print("  All requirements installed.")


def main():
    parser = argparse.ArgumentParser(
        description="Set up the Python environment for the Shirvalkar 2026 deposit.")
    parser.add_argument(
        "--no-venv", action="store_true",
        help="Install into the current Python instead of creating a new "
             "virtual environment (use when you already manage your own "
             "environment, for example an activated conda environment).")
    arguments = parser.parse_args()

    check_python_version()

    if arguments.no_venv:
        # Install straight into whatever Python is running this script.
        print("Installing into the current Python (no virtual environment).")
        install_requirements(sys.executable)
        print("\nDone. You can now run the figure scripts with this Python.")
        return

    create_virtual_environment()
    install_requirements(python_inside_environment())

    print("\nDone. Activate the environment before running the figure scripts:")
    if os.name == "nt":
        print("    %s\\Scripts\\activate" % virtual_environment_directory)
    else:
        print("    source %s/bin/activate" % virtual_environment_directory)
    print("Then, for example:")
    print("    python plot_figureS6_battery_efficiency.py")


if __name__ == "__main__":
    main()
