# find all the build directoris in the src directory and remove them

import os
import shutil
import glob

def clean_build():
    to_delete = ["src/**/build", "src/**/__pycache__"]
    for i in to_delete:
        build_dirs = glob.glob(i, recursive=True)
        for build_dir in build_dirs:
            shutil.rmtree(build_dir)

if __name__ == "__main__":
    clean_build()