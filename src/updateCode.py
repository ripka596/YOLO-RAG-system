import os, shutil, subprocess, stat
from pathlib import Path

REPO_URL, DATA_DIR = "https://github.com/wang-xinyu/tensorrtx.git", Path("data/code_cpp")

ALLOWED_EXT = {
    ".cpp", ".hpp", ".h", ".c", ".cu", ".cuh"
}

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def update_code():
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Git\cmd"
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    shutil.rmtree("temp_yolov8", ignore_errors=True)

    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(DATA_DIR)])

    temp_dir = Path("temp_yolov8")
    shutil.move(str(DATA_DIR / "yolov8"), str(temp_dir))
    shutil.rmtree(DATA_DIR, onexc=remove_readonly)
    shutil.move(str(temp_dir), str(DATA_DIR))

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = Path(root) / file
            is_allowed = (file_path.suffix.lower() in ALLOWED_EXT)

            if not is_allowed:
                file_path.unlink()

    for root, dirs, _ in os.walk(DATA_DIR, topdown=False):
        for dir in dirs:
            dir_path = Path(root) / dir
            if not any(dir_path.iterdir()):
                dir_path.rmdir()

if __name__ == "__main__":
    update_code()