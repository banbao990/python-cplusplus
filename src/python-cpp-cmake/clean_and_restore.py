import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../".format(path))

from utils.clean import clean_build_and_uninstall

# install 
# os.system("pip install -e .") # gen *.pyd at . location
# os.system("pip install .")

if __name__ == "__main__":
    LIB_NAME = "cmake_example"
    clean_build_and_uninstall(LIB_NAME)