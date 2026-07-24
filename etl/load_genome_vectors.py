import sqlite3
import struct
import time
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"

def load_genome_vectors():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    
    cursor = conn.cursor()
    
    # 1. Determine total dimensions (should be 1128)
    cursor.execute("SELECT COUNT(*), MAX(tag_id) FROM genome_tags")
    num_tags, max_tag_id = cursor.fetchone()
    print(f"Detected {num_tags} tags in database, max tag_id: {max_tag_id}")
    
    # We use max_tag_id as the vector size because tag_ids are 1-indexed up to max_tag_id
    vector_size = max_tag_id
    struct_format = f"<{vector_size}f"
    
    # 2. Count movies to process
    cursor.execute("SELECT COUNT(DISTINCT movie_id) FROM genome_scores")
    num_movies = cursor.fetchone()[0]
    print(f"Found {num_movies:,} movies with genome scores in database.")
    
    # 3. Query all scores ordered by movie_id and tag_id
    print("Fetching genome scores and building vectors...")
    cursor.execute("SELECT movie_id, tag_id, relevance FROM genome_scores ORDER BY movie_id, tag_id")
    
    batch_size = 500
    updates = []
    processed_count = 0
    start_time = time.time()
    
    current_movie_id = None
    vector = [0.0] * vector_size
    
    for movie_id, tag_id, relevance in cursor:
        if movie_id != current_movie_id:
            if current_movie_id is not None:
                # Save previous movie vector
                blob = struct.pack(struct_format, *vector)
                updates.append((blob, current_movie_id))
                
                if len(updates) >= batch_size:
                    conn.executemany(
                        "UPDATE movie_embeddings SET v_genome = ?, has_genome = 1 WHERE movie_id = ?", 
                        updates
                    )
                    conn.commit()
                    processed_count += len(updates)
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    print(f"  Processed {processed_count:,}/{num_movies:,} movies ({rate:.1f} movies/sec)")
                    updates = []
            
            # Start new movie
            current_movie_id = movie_id
            vector = [0.0] * vector_size
            
        # Set dimension (tag_id is 1-indexed)
        if 1 <= tag_id <= vector_size:
            vector[tag_id - 1] = float(relevance)
            
    # Process the last movie
    if current_movie_id is not None:
        blob = struct.pack(struct_format, *vector)
        updates.append((blob, current_movie_id))
        conn.executemany(
            "UPDATE movie_embeddings SET v_genome = ?, has_genome = 1 WHERE movie_id = ?", 
            updates
        )
        conn.commit()
        processed_count += len(updates)
        
    print(f"Completed! Loaded {processed_count:,} genome vectors into 'movie_embeddings'.")
    
    # Verify a sample vector
    cursor.execute("SELECT movie_id, length(v_genome) FROM movie_embeddings WHERE has_genome = 1 LIMIT 3")
    samples = cursor.fetchall()
    print("\nVerification check (sample records):")
    for mid, byte_len in samples:
        expected_bytes = vector_size * 4
        status = "OK" if byte_len == expected_bytes else f"FAIL (expected {expected_bytes} bytes, got {byte_len})"
        print(f"  Movie ID {mid}: Vector BLOB size = {byte_len} bytes [{status}]")
        
    conn.close()

if __name__ == "__main__":
    load_genome_vectors()
