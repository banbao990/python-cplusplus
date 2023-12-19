import shutil

def copy_pybind11_full(dst: str):
    SRC = "externals/pybind11"
    shutil.copytree('src/pybind11', dst, dirs_exist_ok=True)


def copy_pybind11_minimal(dst: str):
    pass

if __name__ == '__main__':
    copy_pybind11_full('src/cmake-oidn/pybind11')
    copy_pybind11_full('src/python-cpp-cmake/pybind11')