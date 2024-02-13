""" module adds zhk location into system path and
    improve interaction through pylance in vscode
"""

import sys
from os import makedirs

def create_json(zhk_location: str) -> None:
    """creates a json file to help pylance to recognize ZHK as a package

    Args:
        zhk_location (str): dir of ZHK
    """

    string = r'{"python.analysis.extraPaths": ["' + f"{zhk_location}" + r'"]}'
    makedirs('.vscode', exist_ok=True)
    with open('.vscode/settings.json', 'w', encoding='utf-8') as jsonfile:
        jsonfile.write(string)

def init_zhk(package_dir: str) -> None:
    """ adds json file and adds zhk location into sys.path

    Args:
        package_dir (str): zhk location
    """

    print(package_dir)

    sys.path.insert(1, package_dir)
    create_json(package_dir)
