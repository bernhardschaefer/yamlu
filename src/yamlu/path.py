from pathlib import Path
from typing import List, Iterator, Union

EXCLUDED_FILE_NAMES = {"__MACOSX", "desktop.ini"}


def glob(path: Path, pattern: str, include_hidden=False) -> List[Path]:
    return _get_sorted_paths(path.glob(pattern), include_hidden)


def ls(path: Union[Path, str], include_hidden=False) -> List[Path]:
    if isinstance(path, str):
        path = Path(path)
    return _get_sorted_paths(path.iterdir(), include_hidden)


def _get_sorted_paths(path_iterator: Iterator[Path], include_hidden: bool) -> List[Path]:
    def filter_fn(p: Path):
        if p.name in EXCLUDED_FILE_NAMES:
            return False
        if include_hidden:
            return True
        if p.name.startswith("."):
            return False
        return not any(parent.name.startswith(".") for parent in p.parents)

    return sorted(filter(filter_fn, path_iterator))
