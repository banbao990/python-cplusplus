import os
import sys

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# glob all the install.py files in the subdirectories and run them
import os
import glob


import sys
sys.path.append(os.path.join(CURRENT_DIR, "../"))
from src.config import check_user_settings


def install_all():
    install_files = glob.glob("./**/install.py", recursive=True, root_dir=CURRENT_DIR)
    install_files = [os.path.join(CURRENT_DIR, f) for f in install_files]
    print("    {}".format("\n    ".join(install_files)))
    for install_file in install_files:
        cmd = "python {}".format(install_file)
        print("Running: \033[92m{}\033[00m".format(cmd))
        ret = os.system(cmd)
        if ret != 0:
            print("Failed to run: \033[91m{}\033[00m".format(cmd))
            sys.exit(1)


if __name__ == "__main__":
    check_user_settings()
    install_all()
