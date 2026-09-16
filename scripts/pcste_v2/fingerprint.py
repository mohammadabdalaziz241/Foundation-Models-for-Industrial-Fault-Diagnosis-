"""Print the execution fingerprint for (protocol, fold) on this host without touching a GPU run directory (parity checks)."""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
sys.argv_backup = list(sys.argv)
import importlib.util
spec = importlib.util.spec_from_file_location("runmod", REPO / "scripts/pcste_v2/run.py"); runmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(runmod)
ap = argparse.ArgumentParser(); ap.add_argument("--protocol", default="global_v2_PB_sealedMAF_v1"); ap.add_argument("--fold", type=int, default=1); a = ap.parse_args()
fp = runmod.fingerprint(a.protocol, a.fold, {"probe": True}); fp.pop("config_sha256", None)
print(json.dumps(fp, sort_keys=True))
