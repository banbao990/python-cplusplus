import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../".format(path))

from utils.clean import clean_build_and_uninstall

if __name__ == "__main__":
    lib_name = "python_example"
    clean_build_and_uninstall(lib_name)