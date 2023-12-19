import shutil
import argparse
import os

PYBIND11_SRC = "externals/pybind11"

def copy_pybind11_full(dst: str):
    print("copy all pybind11 files")
    shutil.copytree(PYBIND11_SRC, dst, dirs_exist_ok=True)


def copy_pybind11_minimal(dst: str):
    print("copy minimal pybind11 files")
    necessary_files = [
        os.path.join(PYBIND11_SRC, "include"),
        os.path.join(PYBIND11_SRC, "pybind11"),
        os.path.join(PYBIND11_SRC, "tests"),
        os.path.join(PYBIND11_SRC, "tools"),
        os.path.join(PYBIND11_SRC, "CMakeLists.txt"),
    ]

    for file in necessary_files:
        if os.path.isdir(file):
            shutil.copytree(file, os.path.join(dst, os.path.basename(file)), dirs_exist_ok=True)
        else:
            shutil.copy(file, dst)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--all', action='store_true')
    argparse.add_argument('--clean', action='store_true')
    args = argparse.parse_args()

    dst_dirs = [
        'src/cmake-oidn/pybind11',
        'src/python-cpp-cmake/pybind11'
    ]

    # clean
    if args.clean:
        for dst in dst_dirs:
            shutil.rmtree(dst, ignore_errors=True)

    # copy
    for dst in dst_dirs:
        if args.all:
            copy_pybind11_full(dst)
        else:
            copy_pybind11_minimal(dst)