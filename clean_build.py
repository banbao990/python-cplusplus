# find all the build directoris in the src directory and remove them

import os
import shutil
import glob

def clean_build():
    to_delete = [
        "src/**/build", "src/**/__pycache__", "src/**/dist", "build", "src/**/*.egg-info"
    ]
    for i in to_delete:
        file = glob.glob(i, recursive=True)
        for file in file:
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)

    # find all clean_and_restore.py files and run them
    clean_and_restore_files = glob.glob("src/**/clean_and_restore.py",
                                        recursive=True)
    for clean_and_restore_file in clean_and_restore_files:
        os.system("python {}".format(clean_and_restore_file))

if __name__ == "__main__":
    clean_build()
