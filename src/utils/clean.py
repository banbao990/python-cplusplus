import os
import shutil
import glob

def clean_build_and_uninstall(lib_name: str):
    # clean
    clear_dirs = ["build", "dist", "{}.egg-info".format(lib_name)]
    for dir in clear_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)

    clean_files = glob.glob("*.pyd")
    for file in clean_files:
        os.remove(file)

    # uninstall
    os.system("pip uninstall {}".format(lib_name))
