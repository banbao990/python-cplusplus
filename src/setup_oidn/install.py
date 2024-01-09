import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# download oidn
cmd = "python {}/download_resources.py".format(CURRENT_DIR)
os.system(cmd)

# gen *.pyd at . location
cmd = "pip install -e {}".format(CURRENT_DIR)
# cmd = "pip install {}".format(CURRENT_DIR)

os.system(cmd)
