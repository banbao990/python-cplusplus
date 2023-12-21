import os
import shutil
import glob

def clean_build_and_uninstall(lib_name: str, root_dir: str):
    # clean
    clear_dirs = ["build", "dist", "{}.egg-info".format(lib_name)]
    for dir in clear_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)

    clean_files = glob.glob("*.pyd", root_dir=root_dir)
    clean_files.extend(glob.glob("*.so", root_dir=root_dir))
    for file in clean_files:
        os.remove(os.path.join(root_dir, file))

    # uninstall
    os.system("pip uninstall {}".format(lib_name))
