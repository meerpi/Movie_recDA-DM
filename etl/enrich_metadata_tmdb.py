#!/usr/bin/env python3
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
MAX_WORKERS = 25
BATCH_SIZE = 500

def fetch_tmdb_metadata(item):
    movie_id, tmdb_id, title = item
    url = f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits,release_dates'
    req = urllib.request.Request(url, headers={'User-Agent': 'CineVault/1.0'})
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            
            directors = [c.get('name') for c in data.get('credits', {}).get('crew', []) if c.get('job') == 'Director' and c.get('name')]
            directors_str = ', '.join(directors) if directors else ''
            
            cast = [c.get('name') for c in data.get('credits', {}).get('cast', [])[:5] if c.get('name')]
            actors_str = ', '.join(cast) if cast else ''
            
            mpaa = ''
            for res in data.get('release_dates', {}).get('results', []):
                if res.get('iso_3166_1') == 'US':
                    for rd in res.get('release_dates', []):
                        cert = rd.get('certification')
                        if cert:
                            mpaa = cert
                            break
            
            return {
                'movie_id': movie_id,
                'tmdb_id': tmdb_id,
                'directors': directors_str,
                'actors': actors_str,
                'content_rating': mpaa,
                'directors_list': directors,
                'actors_list': cast,
                'success': True
            }
    except Exception as e:
        return {'movie_id': movie_id, 'tmdb_id': tmdb_id, 'error': str(e), 'success': False}

def run_enrichment(limit=None):
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = 'SELECT movie_id, tmdb_id, title FROM movies WHERE tmdb_id IS NOT NULL AND (directors IS NULL OR directors = "" OR actors IS NULL OR actors = "")'
    if limit:
        query += f' LIMIT {limit}'
    
    cur.execute(query)
    to_fetch = cur.fetchall()
    conn.close()

    print(f'🚀 Starting TMDb Metadata Enrichment for {len(to_fetch):,} movies (Workers: {MAX_WORKERS})...')
    if not to_fetch:
        print('✅ All movies already enriched!')
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
                batch.append((res['content_rating'], res['directors'], res['actors'], res['movie_id']))
            else:
                errors += 1

            if len(batch) >= BATCH_SIZE:
                cur.executemany('UPDATE movies SET content_rating = ?, directors = ?, actors = ? WHERE movie_id = ?', batch)
                conn.commit()
                batch = []

            if completed % 1000 == 0 or completed == len(to_fetch):
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

    print('📄 Syncing updated metadata into tier_c_profile_cards.jsonl...')
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
                if meta['directors_list']:
                    card['directors'] = meta['directors_list']
                if meta['actors_list']:
                    card['actors'] = meta['actors_list']
                if meta['content_rating']:
                    card['content_rating'] = meta['content_rating']
                updated_count += 1
            outfile.write(json.dumps(card, ensure_ascii=False) + chr(10))

    temp_output.replace(TIER_C_CARDS)
    print(f'  ✓ Synced {updated_count:,} profile cards in {TIER_C_CARDS.name}.')

if __name__ == '__main__':
    import sys
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_enrichment(limit=limit_arg)
