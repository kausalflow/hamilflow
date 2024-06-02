from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hamilflow.models.brownian_motion import BrownianMotion


@dataclass
class Dispatcher:
    brownian_motion = BrownianMotion

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())
