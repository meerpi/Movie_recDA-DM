#!/usr/bin/env python3
import os
import re
import csv
import json
import sqlite3
import html

# Filenames
CANDIDATES_CSV = "top_10000_movies.csv"
DESCRIPTIONS_JSONL = "movie_descriptions.jsonl"
METADATA_JSONL = "imdb_metadata_results.jsonl"
REVIEWS_JSONL = "imdb_user_reviews.jsonl"
DB_FILE = "db/cinevault.db"
OUTPUT_JSONL = "tier_a_movies.jsonl"

def print_first_records(filename, n=2):
    print(f"\n--- First {n} records of {filename} ---")
    if not os.path.exists(filename):
        print(f"File {filename} does not exist!")
        return
    with open(filename, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if not line.strip():
                continue
            if count >= n:
                break
            print(line.strip()[:1000] + ("..." if len(line.strip()) > 1000 else ""))
            count += 1
    print()

def normalize_title(title):
    if not title:
        return ""
    title = title.lower().strip()
    
    # Strip leading articles
    for article in ["the ", "a ", "an "]:
        if title.startswith(article):
            title = title[len(article):].strip()
            break
            
    # Strip trailing articles
    for suffix in [", the", ", a", ", an", " the", " a", " an"]:
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()
            break
            
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

def get_variants(title):
    variants = set()
    if not title:
        return variants
    
    variants.add(normalize_title(title))
    
    if '(' in title:
        parts = title.split('(')
        main_title = parts[0].strip()
        variants.add(normalize_title(main_title))
        
        paren_content = parts[1].split(')')[0].strip()
        if paren_content.lower().startswith("a.k.a."):
            aka_title = paren_content[6:].strip()
            variants.add(normalize_title(aka_title))
        else:
            variants.add(normalize_title(paren_content))
            
    return [v for v in variants if v]

def parse_comma_list(val):
    if not val:
        return []
    if isinstance(val, list):
        return [str(item).strip() for item in val if str(item).strip()]
    if isinstance(val, str):
        return [item.strip() for item in val.split(',') if item.strip()]
    return []

def select_top_5_reviews(reviews, title):
    if not reviews:
        return []
    
    def get_sort_key(r):
        votes = r.get("votes", {})
        upvotes = votes.get("up", 0) if isinstance(votes, dict) else 0
        date_str = r.get("date") or ""
        return (upvotes, date_str)
        
    sorted_reviews = sorted(reviews, key=get_sort_key, reverse=True)
    
    if len(sorted_reviews) > 5:
        print(f"[LOG] Truncated audience reviews for '{title}' from {len(sorted_reviews)} to 5.")
        
    top_5 = sorted_reviews[:5]
    
    formatted = []
    for r in top_5:
        summary = r.get("summary") or ""
        text = r.get("text") or ""
        full_text = f"{summary}\n\n{text}".strip() if summary and text else (text or summary)
        
        rating_val = r.get("rating")
        try:
            rating_float = float(rating_val) if rating_val is not None else None
        except (ValueError, TypeError):
            rating_float = None
            
        formatted.append({
            "text": html.unescape(full_text),
            "rating": rating_float
        })
    return formatted

def main():
    # Step 1: Print first 2 records of each JSONL file to confirm structure
    print_first_records(METADATA_JSONL, 2)
    print_first_records(DESCRIPTIONS_JSONL, 2)
    print_first_records(REVIEWS_JSONL, 2)

    # Step 2: Load candidates from CSV
    print("Loading candidate movies from CSV...")
    candidates = []
    if not os.path.exists(CANDIDATES_CSV):
        print(f"Error: {CANDIDATES_CSV} not found!")
        return

    with open(CANDIDATES_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append({
                'movie_id': int(row['movie_id']),
                'title': row['title'],
                'year': int(row['year']) if row['year'] else None,
                'num_ratings': int(row['num_ratings']) if row['num_ratings'] else 0
            })
    print(f"Loaded {len(candidates)} candidate movies from CSV.\n")

    # Step 3: Load movie descriptions
    print("Loading movie descriptions...")
    desc_by_id = {}
    desc_by_title_year = {}
    
    if os.path.exists(DESCRIPTIONS_JSONL):
        with open(DESCRIPTIONS_JSONL, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception as e:
                    print(f"Error parsing line {line_num} in {DESCRIPTIONS_JSONL}: {e}")
                    continue
                
                mid = record.get('movie_id')
                if mid is not None:
                    desc_by_id[int(mid)] = record
                
                title = record.get('title')
                year = record.get('year')
                if title:
                    norm_t = normalize_title(title)
                    try:
                        y_val = int(year) if year is not None else None
                    except (ValueError, TypeError):
                        y_val = None
                    desc_by_title_year[(norm_t, y_val)] = record
    else:
        print(f"Warning: {DESCRIPTIONS_JSONL} not found!")

    # Step 4: Load IMDB metadata
    print("Loading IMDB metadata...")
    meta_by_id = {}
    meta_by_title_year = {}
    
    if os.path.exists(METADATA_JSONL):
        with open(METADATA_JSONL, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception as e:
                    print(f"Error parsing line {line_num} in {METADATA_JSONL}: {e}")
                    continue
                
                mid = record.get('movie_id')
                if mid is not None:
                    meta_by_id[int(mid)] = record
                
                title = record.get('title')
                year = record.get('year')
                if title:
                    norm_t = normalize_title(title)
                    try:
                        y_val = int(year) if year is not None else None
                    except (ValueError, TypeError):
                        y_val = None
                    meta_by_title_year[(norm_t, y_val)] = record
    else:
        print(f"Warning: {METADATA_JSONL} not found!")

    # Step 5: Load scraped audience reviews
    print("Loading scraped audience reviews...")
    audience_by_id = {}
    audience_by_title_year = {}
    
    if os.path.exists(REVIEWS_JSONL):
        with open(REVIEWS_JSONL, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception as e:
                    print(f"Error parsing line {line_num} in {REVIEWS_JSONL}: {e}")
                    continue
                
                mid = record.get('movie_id')
                if mid is not None:
                    audience_by_id[int(mid)] = record
                
                title = record.get('title')
                year = record.get('year')
                if title:
                    norm_t = normalize_title(title)
                    try:
                        y_val = int(year) if year is not None else None
                    except (ValueError, TypeError):
                        y_val = None
                    audience_by_title_year[(norm_t, y_val)] = record
    else:
        print(f"Warning: {REVIEWS_JSONL} not found!")

    # Step 6: Load DB details (critic reviews and genres)
    print("Loading database information...")
    db_by_id = {}
    db_by_title_year = {}
    genres_by_id = {}
    critic_reviews_by_id = {}

    if os.path.exists(DB_FILE):
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            cursor.execute("SELECT movie_id, title, year, directors, actors, writers FROM movies")
            for mid, title, year, directors, actors, writers in cursor.fetchall():
                db_by_id[mid] = {
                    'movie_id': mid,
                    'title': title,
                    'year': year,
                    'directors': directors,
                    'actors': actors,
                    'writers': writers
                }
                if title:
                    norm_t = normalize_title(title)
                    db_by_title_year[(norm_t, year)] = mid
            
            cursor.execute("SELECT mg.movie_id, g.name FROM movie_genres mg JOIN genres g ON mg.genre_id = g.genre_id")
            for mid, genre_name in cursor.fetchall():
                if mid not in genres_by_id:
                    genres_by_id[mid] = []
                genres_by_id[mid].append(genre_name)
                
            cursor.execute("SELECT movie_id, review_text FROM reviews WHERE source IN ('ebert', 'rogerebert')")
            for mid, r_text in cursor.fetchall():
                if r_text and r_text.strip():
                    if mid not in critic_reviews_by_id:
                        critic_reviews_by_id[mid] = r_text.strip()
                        
            conn.close()
        except Exception as e:
            print(f"Error querying database: {e}")
    else:
        print(f"Warning: Database at {DB_FILE} not found!")

    # Step 7: Perform joining and Tier A check
    print("Processing and joining candidate movies...")
    tier_a_movies = []
    
    # Exclusions counters
    excluded_missing_review = 0
    excluded_missing_plot = 0
    excluded_missing_genre = 0
    excluded_missing_director_cast_writer = 0
    
    # Statistic counters
    stat_critic_only = 0
    stat_audience_only = 0
    stat_both = 0
    
    for cand in candidates:
        cand_id = cand['movie_id']
        cand_title = cand['title']
        cand_year = cand['year']
        cand_ratings = cand['num_ratings']
        
        # --- 1. Database Match (for genres and critic review) ---
        db_mid = None
        if cand_id in db_by_id:
            db_mid = cand_id
        else:
            variants = get_variants(cand_title)
            for var in variants:
                for y in [cand_year, cand_year - 1, cand_year + 1] if cand_year else [None]:
                    if (var, y) in db_by_title_year:
                        db_mid = db_by_title_year[(var, y)]
                        break
                if db_mid is not None:
                    break
            if db_mid is None:
                print(f"[LOG] Candidate movie {cand_id} ({cand_title}, {cand_year}) failed to match on DB movies table")
        
        genres = []
        critic_review = None
        if db_mid is not None:
            genres = genres_by_id.get(db_mid, [])
            critic_review = critic_reviews_by_id.get(db_mid)
            
        # --- 2. Movie Descriptions Match (for plot) ---
        desc_rec = None
        if cand_id in desc_by_id:
            desc_rec = desc_by_id[cand_id]
        else:
            variants = get_variants(cand_title)
            for var in variants:
                for y in [cand_year, cand_year - 1, cand_year + 1] if cand_year else [None]:
                    if (var, y) in desc_by_title_year:
                        desc_rec = desc_by_title_year[(var, y)]
                        break
                if desc_rec is not None:
                    break
            if desc_rec is None:
                print(f"[LOG] Candidate movie {cand_id} ({cand_title}, {cand_year}) failed to match on {DESCRIPTIONS_JSONL}")
                
        plot = None
        if desc_rec is not None:
            plot = desc_rec.get('plot')
            if not plot or not plot.strip():
                plot = desc_rec.get('intro')
            if plot:
                plot = plot.strip()
                
        # --- 3. IMDB Metadata Match (for director, cast, writer) ---
        meta_rec = None
        if cand_id in meta_by_id:
            meta_rec = meta_by_id[cand_id]
        else:
            variants = get_variants(cand_title)
            for var in variants:
                for y in [cand_year, cand_year - 1, cand_year + 1] if cand_year else [None]:
                    if (var, y) in meta_by_title_year:
                        meta_rec = meta_by_title_year[(var, y)]
                        break
                if meta_rec is not None:
                    break
            if meta_rec is None:
                print(f"[LOG] Candidate movie {cand_id} ({cand_title}, {cand_year}) failed to match on {METADATA_JSONL}")
                
        # Load metadata from DB if matched
        db_rec = db_by_id.get(db_mid) if db_mid is not None else None
        db_director = parse_comma_list(db_rec.get('directors')) if db_rec else []
        db_cast = parse_comma_list(db_rec.get('actors')) if db_rec else []
        db_writer = parse_comma_list(db_rec.get('writers')) if db_rec else []
        
        # Load from JSONL as fallback/merge
        jsonl_director = parse_comma_list(meta_rec.get('directors')) if meta_rec else []
        jsonl_cast = parse_comma_list(meta_rec.get('actors')) if meta_rec else []
        jsonl_writer = parse_comma_list(meta_rec.get('writers')) if meta_rec else []
        
        director = db_director if db_director else jsonl_director
        cast = db_cast if db_cast else jsonl_cast
        writer = db_writer if db_writer else jsonl_writer
            
        # --- 4. IMDb User Reviews Match ---
        audience_rec = None
        if cand_id in audience_by_id:
            audience_rec = audience_by_id[cand_id]
        else:
            variants = get_variants(cand_title)
            for var in variants:
                for y in [cand_year, cand_year - 1, cand_year + 1] if cand_year else [None]:
                    if (var, y) in audience_by_title_year:
                        audience_rec = audience_by_title_year[(var, y)]
                        break
                if audience_rec is not None:
                    break
            if audience_rec is None:
                print(f"[LOG] Candidate movie {cand_id} ({cand_title}, {cand_year}) failed to match on {REVIEWS_JSONL}")
                
        raw_audience_reviews = []
        if audience_rec is not None:
            raw_audience_reviews = audience_rec.get('reviews') or []
            
        # Select and format top 5 reviews
        audience_reviews = select_top_5_reviews(raw_audience_reviews, cand_title)

        # --- Tier A Checks ---
        
        # 1. At least one review source: ebert or audience
        has_critic = critic_review is not None
        has_audience = len(audience_reviews) > 0
        if not (has_critic or has_audience):
            excluded_missing_review += 1
            print(f"[LOG] Candidate movie {cand_id} ({cand_title}, {cand_year}) excluded: missing reviews.")
            continue
            
        # 2. Plot summary present
        if not plot:
            excluded_missing_plot += 1
            print(f"[LOG] Candidate movie {cand_id} ({cand_title}, {cand_year}) excluded: missing plot summary.")
            continue
            
        # 3. At least one genre
        if not genres:
            excluded_missing_genre += 1
            print(f"[LOG] Candidate movie {cand_id} ({cand_title}, {cand_year}) excluded: missing genres.")
            continue
            
        # 4. At least one of: director, cast, writer
        if not (director or cast or writer):
            excluded_missing_director_cast_writer += 1
            print(f"[LOG] Candidate movie {cand_id} ({cand_title}, {cand_year}) excluded: missing director, cast, and writer.")
            continue
            
        # Movie qualifies for Tier A!
        tier_a_movies.append({
            "movie_id": cand_id,
            "title": cand_title,
            "year": cand_year,
            "num_ratings": cand_ratings,
            "plot": plot,
            "critic_review": critic_review,
            "audience_reviews": audience_reviews,
            "genres": genres,
            "director": director,
            "cast": cast,
            "writer": writer,
            "tier": "A"
        })
        
        # Statistics compilation
        if has_critic and has_audience:
            stat_both += 1
        elif has_critic:
            stat_critic_only += 1
        else:
            stat_audience_only += 1

    # Step 8: Sort and Write to Output
    print(f"\nSorting {len(tier_a_movies)} qualified movies by num_ratings descending...")
    tier_a_movies.sort(key=lambda x: x['num_ratings'], reverse=True)

    print(f"Writing to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for m in tier_a_movies:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
            
    # Step 9: Print Validation Report
    print("\n" + "="*50)
    print("VALIDATION REPORT")
    print("="*50)
    print(f"Total candidate movies checked:               {len(candidates)}")
    print(f"Total qualified as Tier A:                    {len(tier_a_movies)}")
    
    print("\nExclusions Breakdown:")
    print(f"  - Missing both critic and audience reviews: {excluded_missing_review}")
    print(f"  - Missing plot summary:                     {excluded_missing_plot}")
    print(f"  - Missing genres:                           {excluded_missing_genre}")
    print(f"  - Missing all metadata (director/cast/writ): {excluded_missing_director_cast_writer}")
    
    print("\nQualifying Tier A Reviews Breakdown:")
    print(f"  - Critic review only:                       {stat_critic_only}")
    print(f"  - Audience review only:                     {stat_audience_only}")
    print(f"  - Both critic and audience reviews:         {stat_both}")
    
    critic_reviewed_count = stat_critic_only + stat_both
    audience_reviewed_count = stat_audience_only + stat_both
    overlap_percentage = (stat_both / critic_reviewed_count * 100) if critic_reviewed_count > 0 else 0.0
    
    print("\nCritic & Audience Overlap Analysis:")
    print(f"  - Critic-reviewed set size:                 {critic_reviewed_count}")
    print(f"  - Audience-reviewed set size:               {audience_reviewed_count}")
    print(f"  - Overlap count (in both):                  {stat_both}")
    print(f"  - Overlap percentage of critic set:         {overlap_percentage:.2f}%")
    
    print("\n" + "="*50)
    print("Eyeball Verification (First 3 output records):")
    print("="*50)
    print_first_records(OUTPUT_JSONL, 3)

if __name__ == "__main__":
    main()
