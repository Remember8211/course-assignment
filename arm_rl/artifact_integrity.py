"""Portable checks for historical inputs; no model/environment dependency."""
from pathlib import Path, PureWindowsPath
import hashlib

OPTIONAL_PAPER = 'paper/论文_机械臂版_v3.docx'

def normalized_relative(value):
    path = PureWindowsPath(str(value))
    if path.drive or path.root or '..' in path.parts:
        raise ValueError(f'Expected a repository-relative path: {value}')
    return Path(*path.parts)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def snapshot_original_artifacts(repository, relatives):
    repository = Path(repository)
    hashes, omitted = {}, []
    for relative in relatives:
        relative = normalized_relative(relative)
        path = repository / relative
        if relative.as_posix() == OPTIONAL_PAPER and not path.is_file():
            omitted.append(relative.as_posix())
            continue
        hashes[relative.as_posix()] = digest(path)
    return hashes, omitted

def verify_original_artifacts(repository, expected):
    repository = Path(repository)
    verified, not_checked = [], []
    for relative, wanted in expected.items():
        relative = normalized_relative(relative)
        path = repository / relative
        if relative.as_posix() == OPTIONAL_PAPER and not path.is_file():
            not_checked.append(relative.as_posix())
            continue
        if digest(path) != wanted:
            raise ValueError(f'Historical artifact hash mismatch: {relative.as_posix()}')
        verified.append(relative.as_posix())
    if not_checked:
        print('Optional manuscript not checked (not distributed): ' + ', '.join(not_checked))
    return {'verified': verified, 'not_checked_optional': not_checked,
            'all_required_verified': True, 'all_listed_verified': not not_checked}
