import os, sys

def clear():
    """terminal clear"""
    os.system('cls' if sys.platform == 'win32' else 'clear')