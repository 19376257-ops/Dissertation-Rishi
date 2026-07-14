from pathlib import Path
import json

class DataManager:
    def __init__(self, root_dir: Path):
        self.root = root_dir

    def get_paths(self, iteration: str, protocol: str, date: str, round_: str) -> dict:
        base = self.root / iteration / protocol / date / round_
        return {
            'raw':    base / 'raw',     # where input files live
            'results': base / 'results'  # where to save outputs
        }


    def load_experiment_params(self, raw_dir: Path) -> dict:
        local = raw_dir.parent / "experiment_params.json"
        if local.exists():
            path = local
        else:
            path = self.root / "experiment_params.json"
        with open(path, "r") as f:
            return json.load(f)