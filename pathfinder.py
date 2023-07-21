import sys
from pathlib import Path

def create_json(ZHK_location: str):
    string = r'{"python.analysis.extraPaths": ["' + f"{ZHK_location}" + r'"]}'
    with open('.vscode/settings.json', 'w', encoding='utf-8') as f:
        f.write(string)

def init_ZHK():
    cwd = Path.cwd()

    if sys.platform == "darwin":
        # OS X
        ZHK_location = str(Path(*cwd.parts[:3])) + r'/lib'

    elif sys.platform == "win32":
        # Windows
        ZHK_location = Path.as_posix(Path(cwd.anchor + r'lib'))
    else:
        pass
    
    print(ZHK_location)

    sys.path.insert(1, ZHK_location)
    create_json(ZHK_location)

    return str(ZHK_location)