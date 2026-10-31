import shutil, subprocess, os, stat
from pathlib import Path
from ingest_bm25 import build_bm25
from ingest_data import update_chromaDB

DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "docs"
CPP_DIR = DATA_DIR / "code_cpp"
PY_DIR = DATA_DIR / "code_py"

MAIN_REPO_URL = "https://github.com/ultralytics/ultralytics.git"
CPP_REPO_URL = "https://github.com/wang-xinyu/tensorrtx.git"
CPP_EXT = {".cpp", ".hpp", ".h", ".c", ".cu", ".cuh"}

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Git\cmd"

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    func(path)

def update_data():
    shutil.rmtree(DOCS_DIR, ignore_errors=True)
    shutil.rmtree(PY_DIR, ignore_errors=True)
    shutil.rmtree("temp_docs", ignore_errors=True)

    subprocess.run(["git", "clone", "--depth", "1", MAIN_REPO_URL, str(DOCS_DIR)])

    temp_dir, code_dir = Path("temp_docs"), PY_DIR
    shutil.move(str(DOCS_DIR / "ultralytics"), str(code_dir))
    shutil.move(str(DOCS_DIR / "docs"/ "en"), str(temp_dir))
    shutil.rmtree(DOCS_DIR, onexc=remove_readonly)
    shutil.move(str(temp_dir), str(DOCS_DIR))


def update_code_cpp():
    shutil.rmtree(CPP_DIR, ignore_errors=True)
    shutil.rmtree("temp_tensorrtx", ignore_errors=True)

    subprocess.run(["git", "clone", "--depth", "1", CPP_REPO_URL, str(CPP_DIR)])

    temp_dir = Path("temp_tensorrtx")
    shutil.move(str(CPP_DIR / "yolov8"), str(temp_dir))
    shutil.rmtree(CPP_DIR, onexc=remove_readonly)
    shutil.move(str(temp_dir), str(CPP_DIR))

    for root, _, files in os.walk(CPP_DIR):
        for file in files:
            file_path = Path(root) / file
            is_allowed = (file_path.suffix.lower() in CPP_EXT)

            if not is_allowed:
                file_path.unlink()

    for root, dirs, _ in os.walk(CPP_DIR, topdown=False):
        for dir in dirs:
            dir_path = Path(root) / dir
            if not any(dir_path.iterdir()):
                dir_path.rmdir()

def update_code_py():
    for root, _, files in os.walk(PY_DIR):
        for file in files:
            file_path = Path(root) / file
            is_allowed = (file_path.suffix.lower() == ".py" and file != "__init__.py")

            if not is_allowed:
                file_path.unlink()

    for root, dirs, _ in os.walk(PY_DIR, topdown=False):
        for dir in dirs:
            dir_path = Path(root) / dir
            if not any(dir_path.iterdir()):
                dir_path.rmdir()


if __name__ == "__main__":
    update_data()
    update_code_cpp()
    update_code_py()
    update_chromaDB()
    build_bm25()
