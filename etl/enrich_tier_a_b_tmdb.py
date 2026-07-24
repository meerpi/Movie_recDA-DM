#!/usr/bin/env python3
"""
etl/enrich_tier_a_b_tmdb.py — TMDb Metadata Enrichment for Tier A and Tier B Profile Cards

Fetches rich metadata for Tier A and Tier B movies using TMDb API:
 - directors, actors (top 5 cast roster), content_rating (MPAA)
 - original_language, production_countries, collection, poster_path, backdrop_path, tagline
 - Tier A: SKIPS keywords (preserves rich top_tags & LLM review fields)
 - Tier B: INCLUDES keywords (supplements thin tag genome)

Updates both SQLite db/cinevault.db and JSONL profile cards:
 - tier_a_profile_cards_v3.jsonl
 - tier_b_profile_cards.jsonl
"""

import json
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / 'db' / 'cinevault.db'
TIER_A_CARDS = PROJECT_ROOT / 'tier_a_profile_cards_v3.jsonl'
TIER_B_CARDS = PROJECT_ROOT / 'tier_b_profile_cards.jsonl'

TMDB_API_KEY = '2cc1fd2a583e2e14f6b634fb124f9ced'
MAX_WORKERS = 40
BATCH_SIZE = 500


def load_card_movie_ids(card_path: Path) -> set:
    """Loads movie_ids present in a jsonl profile card file."""
    ids = set()
    if not card_path.exists():
        return ids
    with open(card_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    card = json.loads(line)
                    ids.add(int(card['movie_id']))
                except Exception:
                    pass
    return ids


def fetch_tmdb_metadata(item):
    movie_id, tmdb_id, title = item
    url = f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits,keywords,release_dates'
    req = urllib.request.Request(url, headers={'User-Agent': 'CineVault/1.0'})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))

            # Directors & Cast
            directors = [
                c.get('name')
                for c in data.get('credits', {}).get('crew', [])
                if c.get('job') == 'Director' and c.get('name')
            ]
            cast = [
                c.get('name')
                for c in data.get('credits', {}).get('cast', [])[:5]
                if c.get('name')
            ]

            # MPAA Rating (US certification)
            mpaa = ''
            for res in data.get('release_dates', {}).get('results', []):
                if res.get('iso_3166_1') == 'US':
                    for rd in res.get('release_dates', []):
                        cert = rd.get('certification')
                        if cert:
                            mpaa = cert
                            break

            # Extra TMDb fields
            orig_lang = data.get('original_language', '')
            countries = [
                c.get('name')
                for c in data.get('production_countries', [])
                if c.get('name')
            ]
            keywords = [
                k.get('name')
                for k in data.get('keywords', {}).get('keywords', [])
                if k.get('name')
            ]
            collection_info = data.get('belongs_to_collection')
            collection_name = (
                collection_info.get('name')
                if isinstance(collection_info, dict)
                else None
            )
            poster_path = data.get('poster_path')
            backdrop_path = data.get('backdrop_path')
            tagline = data.get('tagline') or ''

            return {
                'movie_id': movie_id,
                'tmdb_id': tmdb_id,
                'directors': directors,
                'actors': cast,
                'content_rating': mpaa,
                'original_language': orig_lang,
                'production_countries': countries,
                'keywords': keywords,
                'collection': collection_name,
                'poster_path': poster_path,
                'backdrop_path': backdrop_path,
                'tagline': tagline,
                'success': True,
            }
    except Exception as e:
        return {
            'movie_id': movie_id,
            'tmdb_id': tmdb_id,
            'error': str(e),
            'success': False,
        }


def run_enrichment(limit=None):
    t0 = time.time()
    tier_a_ids = load_card_movie_ids(TIER_A_CARDS)
    tier_b_ids = load_card_movie_ids(TIER_B_CARDS)

    print(
        f'📄 Loaded {len(tier_a_ids):,} Tier A IDs and {len(tier_b_ids):,} Tier B IDs.'
    )

    all_target_ids = tier_a_ids | tier_b_ids
    if not all_target_ids:
        print('[ERROR] No Tier A or Tier B IDs found.')
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        'SELECT movie_id, tmdb_id, title FROM movies WHERE tmdb_id IS NOT NULL'
    )
    all_movies = cur.fetchall()
    conn.close()

    to_fetch = [m for m in all_movies if m[0] in all_target_ids]
    if limit:
        to_fetch = to_fetch[:limit]

    print(
        f'🚀 Fetching TMDb metadata for {len(to_fetch):,} Tier A/B movies (Workers: {MAX_WORKERS})...'
    )
    if not to_fetch:
        print('✅ No movies to fetch!')
        return

    results = []
    completed = 0
    errors = 0

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_tmdb_metadata, item): item for item in to_fetch
        }

        batch = []
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res.get('success'):
                results.append(res)
                dirs_str = ', '.join(res['directors'])
                acts_str = ', '.join(res['actors'])
                batch.append(
                    (
                        res['content_rating'],
                        dirs_str,
                        acts_str,
                        res['movie_id'],
                    )
                )
            else:
                errors += 1

            if len(batch) >= BATCH_SIZE:
                cur.executemany(
                    'UPDATE movies SET content_rating = ?, directors = ?, actors = ? WHERE movie_id = ?',
                    batch,
                )
                conn.commit()
                batch = []

            if completed % 500 == 0 or completed == len(to_fetch):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f'  Progress: {completed:,}/{len(to_fetch):,} ({completed/len(to_fetch)*100:.1f}%) | {rate:.1f} movies/sec | Errors: {errors}'
                )

        if batch:
            cur.executemany(
                'UPDATE movies SET content_rating = ?, directors = ?, actors = ? WHERE movie_id = ?',
                batch,
            )
            conn.commit()

    conn.close()
    print(
        f'✅ SQLite update completed in {time.time() - t0:.1f}s. Fetched metadata for {len(results):,} movies.'
    )

    update_profile_cards(
        TIER_A_CARDS, results, is_tier_a=True
    )
    update_profile_cards(
        TIER_B_CARDS, results, is_tier_a=False
    )


def update_profile_cards(card_path: Path, enriched_results: list, is_tier_a: bool):
    if not card_path.exists() or not enriched_results:
        return

    tier_name = 'Tier A' if is_tier_a else 'Tier B'
    print(f'📄 Syncing rich metadata into {tier_name} cards ({card_path.name})...')
    meta_map = {r['movie_id']: r for r in enriched_results if r.get('success')}

    updated_count = 0
    temp_output = card_path.with_suffix('.jsonl.tmp')

    with open(card_path, 'r', encoding='utf-8') as infile, open(
        temp_output, 'w', encoding='utf-8'
    ) as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            mid = card.get('movie_id')
            if mid in meta_map:
                meta = meta_map[mid]
                if meta['directors']:
                    card['directors'] = meta['directors']
                if meta['actors']:
                    card['actors'] = meta['actors']
                if meta['content_rating']:
                    card['content_rating'] = meta['content_rating']
                if meta['original_language']:
                    card['original_language'] = meta['original_language']
                if meta['production_countries']:
                    card['production_countries'] = meta['production_countries']
                if meta['collection']:
                    card['collection'] = meta['collection']
                if meta['poster_path']:
                    card['poster_path'] = meta['poster_path']
                if meta['backdrop_path']:
                    card['backdrop_path'] = meta['backdrop_path']
                if meta['tagline']:
                    card['tagline'] = meta['tagline']

                # Key divergence: Tier B gets keywords; Tier A skips keywords
                if not is_tier_a and meta['keywords']:
                    card['keywords'] = meta['keywords']

                updated_count += 1
            outfile.write(json.dumps(card, ensure_ascii=False) + '\n')

    temp_output.replace(card_path)
    print(f'  ✓ Synced {updated_count:,} profile cards in {card_path.name}.')


if __name__ == '__main__':
    import sys

    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_enrichment(limit=limit_arg)
