import subprocess
import time
from pathlib import Path


def download_weights(url: str, dest: Path):
    """Download weights from URL to destination path"""
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)

    # Ensure destination directory exists
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if url.endswith("tar"):
        subprocess.check_call(
            ["pget", "--log-level=WARNING", "-x", url, dest], close_fds=False
        )
    else:
        subprocess.check_call(
            ["pget", "--log-level=WARNING", url, dest], close_fds=False
        )
    print("downloading took: ", time.time() - start)
