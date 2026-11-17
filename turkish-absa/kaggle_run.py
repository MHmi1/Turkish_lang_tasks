"""Kaggle launcher; the default all-stage run uses public data only."""
import os
import subprocess
import sys
from pathlib import Path

if any(arg in ("synth", "--stage=synth") for arg in sys.argv) and not os.environ.get("GEMINI_API_KEY"):
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_1"):
            try:
                key = client.get_secret(name)
            except Exception:
                continue
            if key:
                os.environ["GEMINI_API_KEY"] = key
                print(f"Loaded {name} from Kaggle Secrets")
                break
    except ImportError:
        pass

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (HERE / "run_all.py").exists():
    for p in [Path("/kaggle/working"), *Path("/kaggle/input").rglob("run_all.py")]:
        cand = p.parent if p.name == "run_all.py" else p
        if (cand / "absa" / "config.py").exists():
            HERE = cand
            break

os.environ.setdefault("ABSA_WORK", "/kaggle/working/absa_work" if Path("/kaggle/working").exists() else str(HERE / "work"))
subprocess.run([sys.executable, str(HERE / "run_all.py"), *sys.argv[1:]], check=True)
