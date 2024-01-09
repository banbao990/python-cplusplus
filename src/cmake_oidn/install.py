import os
import shutil
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
from src.config import check_user_settings

check_user_settings()

# copy weights
weight_src_dir = os.path.join(CURRENT_DIR, "../setup_oidn/weights")
weight_dst_dir = os.path.join(CURRENT_DIR, "weights")
if not os.path.exists(weight_dst_dir):
    os.makedirs(weight_dst_dir)
for file in os.listdir(weight_src_dir):
    src = os.path.join(weight_src_dir, file)
    dst = os.path.join(weight_dst_dir, file)
    if os.path.exists(dst):
        continue
    else:
        shutil.copy(src, dst)

# download oidn
cmd = "python {}/download_resources.py".format(CURRENT_DIR)
os.system(cmd)

# gen *.pyd at . location
cmd = "pip install -e {}".format(CURRENT_DIR)
# cmd = "pip install {}".format(CURRENT_DIR)

os.system(cmd)
