import os
import shutil
import glob
# install 
# os.system("pip install -e .")

if __name__ == "__main__":
    # clean
    clear_dirs = ["build", "dist", "python_example.egg-info"]
    for dir in clear_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
    
    clean_files = glob.glob("*.pyd")
    for file in clean_files:
        os.remove(file)

    # uninstall
    os.system("pip uninstall python_example")