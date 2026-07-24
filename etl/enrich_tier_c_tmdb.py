#!/usr/bin/env python3
"""
etl/enrich_tier_c_tmdb.py — TMDb Metadata Enrichment ONLY for Tier C Profile Cards

Fetches rich metadata for Tier C movies using TMDb API:
 - directors, actors (top 5), content_rating (MPAA)
 - original_language, production_countries, keywords (thematic tags)
 - collection (franchise), poster_path, tagline

Updates both SQLite db/cinevault.db and tier_c_profile_cards.jsonl.
"""

import json
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / 'db' / 'cinevault.db'
TIER_C_CARDS = PROJECT_ROOT / 'tier_c_profile_cards.jsonl'

TMDB_API_KEY = '2cc1fd2a583e2e14f6b634fb124f9ced'
MAX_WORKERS = 40
BATCH_SIZE = 500

def load_tier_c_movie_ids():
    """Loads all movie_ids present in tier_c_profile_cards.jsonl."""
    tier_c_ids = set()
    if not TIER_C_CARDS.exists():
        return tier_c_ids
    with open(TIER_C_CARDS, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    card = json.loads(line)
                    tier_c_ids.add(int(card['movie_id']))
                except Exception:
                    pass
    return tier_c_ids

def fetch_tmdb_metadata(item):
    movie_id, tmdb_id, title = item
    url = f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits,keywords,release_dates'
    req = urllib.request.Request(url, headers={'User-Agent': 'CineVault/1.0'})
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            
            # Directors & Cast
            directors = [c.get('name') for c in data.get('credits', {}).get('crew', []) if c.get('job') == 'Director' and c.get('name')]
            cast = [c.get('name') for c in data.get('credits', {}).get('cast', [])[:5] if c.get('name')]
            
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
            countries = [c.get('name') for c in data.get('production_countries', []) if c.get('name')]
            keywords = [k.get('name') for k in data.get('keywords', {}).get('keywords', []) if k.get('name')]
            collection_info = data.get('belongs_to_collection')
            collection_name = collection_info.get('name') if isinstance(collection_info, dict) else None
            poster_path = data.get('poster_path')
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
                'tagline': tagline,
                'success': True
            }
    except Exception as e:
        return {'movie_id': movie_id, 'tmdb_id': tmdb_id, 'error': str(e), 'success': False}

def run_enrichment(limit=None):
    t0 = time.time()
    tier_c_ids = load_tier_c_movie_ids()
    print(f'📄 Loaded {len(tier_c_ids):,} Tier C movie IDs from {TIER_C_CARDS.name}.')
    
    if not tier_c_ids:
        print('[ERROR] No Tier C IDs found.')
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Query Tier C movies that have a tmdb_id
    cur.execute('SELECT movie_id, tmdb_id, title FROM movies WHERE tmdb_id IS NOT NULL')
    all_movies = cur.fetchall()
    conn.close()

    to_fetch = [m for m in all_movies if m[0] in tier_c_ids]
    if limit:
        to_fetch = to_fetch[:limit]

    print(f'🚀 Starting Rich TMDb Enrichment for {len(to_fetch):,} Tier C movies (Workers: {MAX_WORKERS})...')
    if not to_fetch:
        print('✅ No Tier C movies to fetch!')
        return

    results = []
    completed = 0
    errors = 0

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_tmdb_metadata, item): item for item in to_fetch}
        
        batch = []
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res.get('success'):
                results.append(res)
                dirs_str = ', '.join(res['directors'])
                acts_str = ', '.join(res['actors'])
                batch.append((res['content_rating'], dirs_str, acts_str, res['movie_id']))
            else:
                errors += 1

            if len(batch) >= BATCH_SIZE:
                cur.executemany('UPDATE movies SET content_rating = ?, directors = ?, actors = ? WHERE movie_id = ?', batch)
                conn.commit()
                batch = []

            if completed % 500 == 0 or completed == len(to_fetch):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f'  Progress: {completed:,}/{len(to_fetch):,} ({completed/len(to_fetch)*100:.1f}%) | {rate:.1f} movies/sec | Errors: {errors}')

        if batch:
            cur.executemany('UPDATE movies SET content_rating = ?, directors = ?, actors = ? WHERE movie_id = ?', batch)
            conn.commit()

    conn.close()
    print(f'✅ TMDb SQLite update completed in {time.time() - t0:.1f}s. Enriched {len(results):,} movies.')

    update_tier_c_cards(results)

def update_tier_c_cards(enriched_results):
    if not TIER_C_CARDS.exists() or not enriched_results:
        return

    print('📄 Syncing rich metadata into tier_c_profile_cards.jsonl...')
    meta_map = {r['movie_id']: r for r in enriched_results if r.get('success')}

    updated_count = 0
    temp_output = TIER_C_CARDS.with_suffix('.jsonl.tmp')

    with open(TIER_C_CARDS, 'r', encoding='utf-8') as infile, open(temp_output, 'w', encoding='utf-8') as outfile:
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
                if meta['keywords']:
                    card['keywords'] = meta['keywords']
                if meta['collection']:
                    card['collection'] = meta['collection']
                if meta['poster_path']:
                    card['poster_path'] = meta['poster_path']
                if meta['tagline']:
                    card['tagline'] = meta['tagline']
                updated_count += 1
            outfile.write(json.dumps(card, ensure_ascii=False) + chr(10))

    temp_output.replace(TIER_C_CARDS)
    print(f'  ✓ Synced {updated_count:,} profile cards in {TIER_C_CARDS.name}.')

if __name__ == '__main__':
    import sys
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_enrichment(limit=limit_arg)
