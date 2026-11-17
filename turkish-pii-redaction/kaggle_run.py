"""Self-contained Kaggle script (T4 x2): Turkish PII Detection & Redaction.

Copy into a single Kaggle notebook cell or upload the repo folder and run:
  !pip install -q "transformers>=4.48.0,<5.0.0" "huggingface_hub>=0.25.0,<1.0.0" "accelerate>=0.34.0" matplotlib fastapi uvicorn
  !python kaggle_run.py --stage all
"""
import os
import subprocess
import sys
from pathlib import Path

try:
    from kaggle_secrets import UserSecretsClient
    k = UserSecretsClient().get_secret("GEMINI_API_KEY")
    if k:
        os.environ["GEMINI_API_KEY"] = k
        print("Loaded GEMINI_API_KEY from Kaggle Secrets")
except Exception:
    pass

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (HERE / "run_all.py").exists():
    for p in [Path("/kaggle/working"), *Path("/kaggle/input").rglob("run_all.py")]:
        cand = p.parent if p.name == "run_all.py" else p
        if (cand / "run_all.py").exists():
            HERE = cand
            break

os.environ.setdefault("PII_WORK", "/kaggle/working/work" if Path("/kaggle/working").exists() else str(HERE / "work"))
subprocess.run([sys.executable, str(HERE / "run_all.py"), *sys.argv[1:]], check=True)
