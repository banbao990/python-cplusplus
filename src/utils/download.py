import os
import requests
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(CURRENT_DIR, "../../resources")


def check_resources_dir():
    if not os.path.exists(RESOURCES_DIR):
        os.mkdir(RESOURCES_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.print_help()
