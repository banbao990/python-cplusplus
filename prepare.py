import shutil
import argparse
import os
import zipfile

PYBIND11_SRC = "externals/pybind11"


def copy_pybind11_full(dst: str):
    print("copy all pybind11 files")
    print("    external/pybind11 -> {}".format(dst))
    shutil.copytree(PYBIND11_SRC, dst, dirs_exist_ok=True)


def copy_pybind11_minimal(dst: str):
    print("copy minimal pybind11 files")
    print("    external/pybind11 -> {}".format(dst))
    necessary_files = [
        os.path.join(PYBIND11_SRC, "include"),
        os.path.join(PYBIND11_SRC, "pybind11"),
        os.path.join(PYBIND11_SRC, "tests"),
        os.path.join(PYBIND11_SRC, "tools"),
        os.path.join(PYBIND11_SRC, "CMakeLists.txt"),
    ]

    for file in necessary_files:
        if os.path.isdir(file):
            shutil.copytree(file, os.path.join(
                dst, os.path.basename(file)), dirs_exist_ok=True)
        else:
            shutil.copy(file, dst)


def deal_with_pybind11(clean, all):
    print("\033[92mCopying pybind11\033[0m")
    if not os.path.exists(PYBIND11_SRC):
        cmd = "git submodule update --init --recursive"
        print("\033[91mpybind11 Not Found, please run {} in tht root dir\033[0m".format(cmd))
        return

    dst_dirs = [
        'src/cmake-oidn/pybind11',
        "src/csetup_oidn/pybind11"
        'src/python-cpp-cmake/pybind11',
        "src/cmake-optix/pybind11"
    ]

    # clean
    if clean:
        for dst in dst_dirs:
            shutil.rmtree(dst, ignore_errors=True)

    # copy
    for dst in dst_dirs:
        if all:
            copy_pybind11_full(dst)
        else:
            copy_pybind11_minimal(dst)

def deal_with_assets():
    print("\033[92mCopying Assets\033[0m")

    # find
    scenes = []
    src_dir = "assets/scenes/"
    for file in os.listdir(src_dir):
        if file.endswith(".zip"):
            scenes.append(file)

    # unzip
    dst_dir = "assets/ignore/scenes/"
    for scene in scenes:
        scene_src = os.path.join(src_dir, scene)
        scene_dst_dir = os.path.join(dst_dir, scene.split(".")[0])
        if(os.path.exists(scene_dst_dir)):
            continue
        print("    unzip {}".format(scene))
        with zipfile.ZipFile(scene_src, 'r') as zip_ref:
            zip_ref.extractall(dst_dir)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--all', action='store_true',
                          help='copy all pybind11 files')
    argparse.add_argument('--clean', action='store_true',
                          help='clean pybind11 files')
    args = argparse.parse_args()

    deal_with_pybind11(args.clean, args.all)

    deal_with_assets()
