import shutil, subprocess, os
from pathlib import Path
from ingest_bm25 import build_bm25
from ingest import update_chromaDB

REPO_URL, DATA_DIR = "https://github.com/ultralytics/ultralytics.git", Path("data")

def update_data():
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Git\cmd"
    shutil.rmtree(DATA_DIR, ignore_errors=True)

    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(DATA_DIR)])

    TARGET_DIR = DATA_DIR / "docs/en"
    shutil.rmtree(DATA_DIR / ".git", ignore_errors=True)

    if TARGET_DIR.exists():
        shutil.rmtree("docs", ignore_errors=True)
        TARGET_DIR.rename("docs")
        shutil.rmtree(DATA_DIR, ignore_errors=True)

if __name__ == "__main__":
    update_data()
    update_chromaDB()
    build_bm25()
