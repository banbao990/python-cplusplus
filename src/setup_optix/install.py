import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(os.path.join(CURRENT_DIR, "../"))
sys.path.append(os.path.join(CURRENT_DIR, "../../"))

from src.config import check_user_settings
check_user_settings()

# gen *.pyd at . location
cmd = "pip install -e {}".format(CURRENT_DIR)
# cmd = "pip install {}".format(CURRENT_DIR)

os.system(cmd)
