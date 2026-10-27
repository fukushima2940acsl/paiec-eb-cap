"""Bounded-batch validation and trial -> item -> pair aggregation using SQLite."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import unicodedata

import pandas as pd

from constants import SUBJECT_FIELDS

GIB = 2 ** 30
BATCH_SIZE = 16384


class DataInvalid(ValueError):
    """An explicit data-quality rule failed; quarantine this whole benchmark."""


def disk_check(directory, reserve=GIB):
    if shutil.disk_usage(directory).free < reserve:
        raise RuntimeError('Less than the required disk reserve remains; stopping')


def content_hash(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith(('{', '[')):
        try:
            text = json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False,
                              separators=(',', ':'))
        except (ValueError, TypeError):
            pass
    text = ' '.join(unicodedata.normalize('NFKC', text).split())
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def frames(path, columns=None):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns, use_threads=False):
        yield batch.to_pandas()


def schema(path, required):
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    if not set(required) <= set(names):
        raise DataInvalid('missing_columns:' + ','.join(sorted(set(required) - set(names))))
    return names


def identifiers(frame, columns):
    for key in columns:
        if frame[key].isna().any():
            raise DataInvalid('missing_identifier:' + key)
        frame[key] = frame[key].astype(str)
        if frame[key].str.strip().eq('').any():
            raise DataInvalid('blank_identifier:' + key)
    return frame


def ensure_benchmark(frame, bid):
    if 'benchmark_id' in frame and (
            frame.benchmark_id.isna().any() or set(frame.benchmark_id.astype(str)) != {bid}):
        raise DataInvalid('benchmark_id_mismatch')


def connect_index(path):
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA cache_size=-65536')
    conn.execute('PRAGMA temp_store=FILE')
    conn.execute('CREATE TABLE old_hash (hash TEXT, bid TEXT, PRIMARY KEY(hash,bid)) WITHOUT ROWID')
    conn.execute('CREATE TABLE new_hash (hash TEXT, bid TEXT, PRIMARY KEY(hash,bid)) WITHOUT ROWID')
    return conn


def index_old_items(conn, path, fold_map, disk_directory):
    count = blanks = 0
    schema(path, ['benchmark_id', 'item_content'])
    for frame in frames(path, ['benchmark_id', 'item_content']):
        disk_check(disk_directory)
        rows = []
        for bid, content in frame.itertuples(index=False, name=None):
            if str(bid) not in fold_map:
                raise ValueError('Old item benchmark absent from frozen folds')
            hashed = content_hash(content)
            count += 1
            if hashed:
                rows.append((hashed, str(bid)))
            else:
                blanks += 1
        conn.executemany('INSERT OR IGNORE INTO old_hash VALUES (?,?)', rows)
        conn.commit()
    return {'old_item_rows': count, 'old_items_without_content_hash': blanks}


def reset_scratch(conn):
    for name in ('subjects', 'subject_keys', 'items', 'seen_response', 'agg', 'stage'):
        conn.execute('DROP TABLE IF EXISTS ' + name)
    conn.executescript('''
        CREATE TABLE subjects (sid TEXT PRIMARY KEY, skey TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE subject_keys (skey TEXT PRIMARY KEY, signature TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE items (iid TEXT PRIMARY KEY, hash TEXT) WITHOUT ROWID;
        CREATE TABLE seen_response (rid TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE agg (skey TEXT, iid TEXT, total REAL NOT NULL, n INTEGER NOT NULL,
                          PRIMARY KEY(skey,iid)) WITHOUT ROWID;
        CREATE TABLE stage (sid TEXT, iid TEXT, value REAL, rid TEXT);
    ''')


def validate_benchmark(conn, paths, slug, old_folds, family_exclusions, disk_directory):
    disk_check(disk_directory)
    reset_scratch(conn)
    catalogs = pd.read_parquet(paths['benchmarks.parquet'])
    if len(catalogs) != 1:
        raise DataInvalid('catalog_row_count')
    meta = catalogs.iloc[0]
    if any(meta.get(k) != v for k, v in
           [('release', 'public'), ('response_type', 'binary'), ('granularity', 'item')]):
        raise DataInvalid('catalog_not_public_binary_item')
    bid = meta.get('benchmark_id')
    if not isinstance(bid, str) or not bid.strip():
        raise DataInvalid('invalid_benchmark_id')
    if bid in old_folds:
        raise DataInvalid('benchmark_id_collision_with_baseline')
    info = {'slug': slug, 'benchmark_id': bid, 'status': 'valid', 'missing_response_rows': 0,
            'raw_response_rows': 0, 'valid_response_rows': 0}
    subjects = paths['subjects.parquet']
    names = schema(subjects, ['subject_id'])
    wanted = [n for n in ['subject_id', 'benchmark_id', *SUBJECT_FIELDS] if n in names]
    for frame in frames(subjects, wanted):
        disk_check(disk_directory)
        ensure_benchmark(frame, bid)
        identifiers(frame, ['subject_id'])
        for field in SUBJECT_FIELDS:
            if field not in frame:
                frame[field] = ''
            frame[field] = frame[field].where(frame[field].notna(), '').astype(str)
        for row in frame[['subject_id', *SUBJECT_FIELDS]].itertuples(index=False, name=None):
            joined = '\x1f'.join(row[1:])
            skey = hashlib.sha1(joined.encode('utf-8')).hexdigest()[:16]
            signature = json.dumps(row[1:], ensure_ascii=False)
            existing = conn.execute('SELECT signature FROM subject_keys WHERE skey=?', (skey,)).fetchone()
            if existing is not None and existing[0] != signature:
                raise DataInvalid('subject_key_collision')
            conn.execute('INSERT OR IGNORE INTO subject_keys VALUES (?,?)', (skey, signature))
            try:
                conn.execute('INSERT INTO subjects VALUES (?,?)', (row[0], skey))
            except sqlite3.IntegrityError:
                raise DataInvalid('duplicate_subject_id') from None
        conn.commit()
    item_path = paths['items.parquet']
    names = schema(item_path, ['item_id'])
    content = 'content' if 'content' in names else 'item_content' if 'item_content' in names else None
    if content is None:
        raise DataInvalid('missing_item_content_column')
    wanted = [n for n in ['item_id', 'benchmark_id', content] if n in names]
    for frame in frames(item_path, wanted):
        disk_check(disk_directory)
        ensure_benchmark(frame, bid)
        identifiers(frame, ['item_id'])
        rows = [(iid, content_hash(value)) for iid, value in
                frame[['item_id', content]].itertuples(index=False, name=None)]
        try:
            conn.executemany('INSERT INTO items VALUES (?,?)', rows)
        except sqlite3.IntegrityError:
            raise DataInvalid('duplicate_item_id') from None
        conn.commit()
    response_path = paths.get('responses.parquet', paths.get('response.parquet'))
    names = schema(response_path, ['response_id', 'subject_id', 'item_id', 'response'])
    wanted = [n for n in ['response_id', 'subject_id', 'item_id', 'response', 'benchmark_id'] if n in names]
    for frame in frames(response_path, wanted):
        disk_check(disk_directory)
        ensure_benchmark(frame, bid)
        info['raw_response_rows'] += len(frame)
        identifiers(frame, ['response_id', 'subject_id', 'item_id'])
        try:
            conn.executemany('INSERT INTO seen_response VALUES (?)',
                             ((rid,) for rid in frame.response_id))
        except sqlite3.IntegrityError:
            raise DataInvalid('duplicate_response_id') from None
        # Reject malformed non-missing grades, rather than silently coercing them.
        missing = frame.response.isna()
        if not frame.loc[~missing, 'response'].isin([0, 1]).all():
            raise DataInvalid('non_binary_response')
        info['missing_response_rows'] += int(missing.sum())
        conn.execute('DELETE FROM stage')
        conn.executemany('INSERT INTO stage VALUES (?,?,?,?)',
            [(sid, iid, None if pd.isna(value) else float(value), rid)
             for sid, iid, value, rid in frame[['subject_id', 'item_id', 'response', 'response_id']]
             .itertuples(index=False, name=None)])
        for table, column, key in [('subjects', 'sid', 'sid'), ('items', 'iid', 'iid')]:
            if conn.execute(f'SELECT 1 FROM stage LEFT JOIN {table} t ON stage.{key}=t.{column} '
                            f'WHERE t.{column} IS NULL LIMIT 1').fetchone():
                raise DataInvalid('unmatched_' + table)
        conn.execute('''INSERT INTO agg
            SELECT s.skey, x.iid, SUM(x.value), COUNT(*) FROM stage x
            JOIN subjects s ON x.sid=s.sid WHERE x.value IS NOT NULL GROUP BY s.skey,x.iid
            ON CONFLICT(skey,iid) DO UPDATE SET total=agg.total+excluded.total, n=agg.n+excluded.n''')
        info['valid_response_rows'] += int((~missing).sum())
        conn.commit()
        if info['raw_response_rows'] % (BATCH_SIZE * 64) == 0:
            print(f"  {slug}: inspected {info['raw_response_rows']:,} response rows", flush=True)
    if not info['valid_response_rows']:
        raise DataInvalid('no_usable_responses')
    unknown_content = conn.execute('''SELECT COUNT(*) FROM
        (SELECT DISTINCT agg.iid FROM agg JOIN items ON agg.iid=items.iid WHERE items.hash IS NULL)''').fetchone()[0]
    info['observed_items_without_content_hash'] = unknown_content
    if unknown_content:
        raise DataInvalid('observed_items_without_content_for_overlap_check')
    old_overlap = dict(conn.execute('''SELECT o.bid, COUNT(DISTINCT i.hash) FROM items i
        JOIN old_hash o ON i.hash=o.hash GROUP BY o.bid''').fetchall())
    new_overlap = dict(conn.execute('''SELECT o.bid, COUNT(DISTINCT i.hash) FROM items i
        JOIN new_hash o ON i.hash=o.hash GROUP BY o.bid''').fetchall())
    explicit = set(family_exclusions.get(slug, []))
    excluded_folds = sorted({old_folds[b] for b in set(old_overlap) | explicit if b in old_folds})
    info.update(overlap_old_benchmarks=old_overlap, overlap_previous_new_benchmarks=new_overlap,
                explicit_related_old_benchmarks=sorted(explicit), excluded_folds=excluded_folds)
    info['duplicate_content_item_excess'] = conn.execute('''SELECT COALESCE(SUM(n-1),0) FROM
        (SELECT COUNT(*) n FROM items WHERE hash IS NOT NULL GROUP BY hash HAVING COUNT(*)>1)''').fetchone()[0]
    stats = pd.read_sql_query('''SELECT skey AS subject_key, AVG(total/n) AS mean,
        COUNT(*) AS n_items FROM agg GROUP BY skey''', conn)
    stats.insert(0, 'benchmark_id', bid)
    stats['excluded_folds'] = json.dumps(excluded_folds)
    info['n_pairs'] = len(stats)
    info['n_cap_pairs'] = int(stats.n_items.between(80, 1000).sum())
    conn.execute('INSERT OR IGNORE INTO new_hash SELECT DISTINCT hash,? FROM items WHERE hash IS NOT NULL', (bid,))
    conn.commit()
    return stats, info
