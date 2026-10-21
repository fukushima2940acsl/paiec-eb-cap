"""Small helpers for release provenance; no credentials or raw data are exported."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_once(path, value):
    """Idempotent for identical JSON; refuse to silently replace release evidence."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'既存の公開用記録と異なります。差を確認してください: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def link_directory(link, target):
    target = Path(target).resolve(strict=True)
    if not target.is_dir():
        raise ValueError(f'ディレクトリではありません: {target}')
    link = Path(link)
    if link.exists() or link.is_symlink():
        if link.resolve() == target:
            return
        raise ValueError(f'既存パスを置き換えません: {link}')
    link.symlink_to(target, target_is_directory=True)


def archive_contents(path):
    with ZipFile(path) as z:
        if sorted(z.namelist()) != ['model.py', 'prior.json'] or z.testzip() is not None:
            raise ValueError('提出ZIPのルートにはmodel.pyとprior.jsonだけが必要です')
        return {name: z.read(name) for name in z.namelist()}


def member_hashes(contents):
    return {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())}


def choose_archive(source, explicit=None):
    if explicit:
        return Path(explicit).resolve(strict=True)
    candidates = sorted((Path(source) / 'dist').glob('eb_cap_all_*/eb_cap_all.zip'))
    if not candidates:
        raise ValueError('eb_cap_all.zipが見つかりません。--submission-zipで明示してください')
    groups = {}
    for path in candidates:
        groups.setdefault(sha_file(path), []).append(path)
    if len(groups) != 1:
        raise ValueError('候補ZIPが複数あります。提出1008013に使ったものを--submission-zipで指定してください:\n'
                         + '\n'.join(str(p) for p in candidates))
    return candidates[0]
