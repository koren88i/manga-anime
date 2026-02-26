"""
Pass 1 — Dialogue Anchor Establishment

Matches manga dialogue lines to anime subtitle lines using dual embedding
methods (OpenAI + bag-of-words fallback). Produces sparse high-confidence
anchors that constrain Pass 2 gap filling.

Uses dynamic programming with monotonicity constraint to preserve sequential
order — if manga line A comes before B, their anime matches must also be
in order.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import logging
import math
import re
import time

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost tracking (shared pattern with anime_metadata.py)
# ---------------------------------------------------------------------------

API_COSTS = {
    "embedding_input_tokens": 0,
    "embedding_calls": 0,
}

PRICING = {
    "embedding_per_1m": 0.02,  # text-embedding-3-small: $0.02/1M tokens
}


def get_embedding_cost_summary():
    """Return a cost summary string for embedding API calls."""
    cost = (API_COSTS["embedding_input_tokens"] / 1_000_000) * PRICING["embedding_per_1m"]
    return (
        f"Embedding API: {API_COSTS['embedding_calls']} calls, "
        f"{API_COSTS['embedding_input_tokens']:,} tokens, ${cost:.4f}"
    )


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "embedding_model": "text-embedding-3-small",
    "match_threshold": 0.65,
    "speaker_match_bonus": 0.10,
    "location_match_bonus": 0.05,
    "time_context_match_bonus": 0.05,
    "embedding_batch_size": 2048,  # OpenAI max batch size
    "embedding_max_retries": 3,
    "embedding_retry_delay": 5,
}


# ---------------------------------------------------------------------------
# Embedding Method 1: OpenAI text-embedding-3-small
# ---------------------------------------------------------------------------

def _embed_openai(texts, config):
    """
    Embed a list of texts using OpenAI's embedding API.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), or None on failure.
    """
    api_key = config.get("openai_api_key", "")
    if not api_key:
        logger.warning("No OPENAI_API_KEY set -- skipping OpenAI embeddings")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed -- skipping OpenAI embeddings")
        return None

    client = OpenAI(api_key=api_key)
    model = config.get("embedding_model", "text-embedding-3-small")

    all_embeddings = []
    batch_size = config.get("embedding_batch_size", 2048)

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]

        for attempt in range(config.get("embedding_max_retries", 3)):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                )
                # Track costs
                if hasattr(response, "usage") and response.usage:
                    API_COSTS["embedding_input_tokens"] += response.usage.total_tokens
                API_COSTS["embedding_calls"] += 1

                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break

            except Exception as e:
                logger.warning(
                    f"OpenAI embedding error (attempt {attempt + 1}): {e}"
                )
                if attempt < config.get("embedding_max_retries", 3) - 1:
                    time.sleep(config.get("embedding_retry_delay", 5))
                else:
                    logger.warning("OpenAI embeddings failed after all retries")
                    return None

    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding Method 2: Bag-of-words cosine (numpy only, zero external deps)
# ---------------------------------------------------------------------------

def _embed_bow(texts):
    """
    Embed texts using bag-of-words term frequency vectors.

    Returns:
        np.ndarray of shape (len(texts), vocab_size)
    """
    # Tokenize all texts
    tokenized = []
    vocab = {}
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        tokenized.append(words)
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)

    if not vocab:
        return np.zeros((len(texts), 1), dtype=np.float32)

    # Build term-frequency matrix
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, words in enumerate(tokenized):
        for w in words:
            matrix[i, vocab[w]] += 1.0

    # L2-normalize each row (so dot product = cosine similarity)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    matrix = matrix / norms

    return matrix


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(embeddings_a, embeddings_b):
    """
    Compute cosine similarity matrix between two sets of L2-normalized embeddings.

    Returns:
        np.ndarray of shape (len(a), len(b))
    """
    # L2-normalize (in case they aren't already)
    norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norms_a[norms_a == 0] = 1.0
    a_normed = embeddings_a / norms_a

    norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)
    norms_b[norms_b == 0] = 1.0
    b_normed = embeddings_b / norms_b

    return a_normed @ b_normed.T


def _apply_metadata_bonuses(sim_matrix, manga_dialogue, anime_dialogue, config):
    """
    Add bonuses to similarity scores based on metadata agreement.

    Modifies sim_matrix in place.
    """
    speaker_bonus = config.get("speaker_match_bonus", 0.10)
    location_bonus = config.get("location_match_bonus", 0.05)
    time_bonus = config.get("time_context_match_bonus", 0.05)

    for i, m_line in enumerate(manga_dialogue):
        m_speaker = m_line.get("speaker_id") or ""
        m_location = (m_line.get("location") or "").lower().strip()
        m_time = (m_line.get("time_context") or "").lower().strip()

        for j, a_line in enumerate(anime_dialogue):
            a_speaker = a_line.get("speaker_id") or ""
            a_location = (a_line.get("location") or "").lower().strip()
            a_time = (a_line.get("time_context") or "").lower().strip()

            bonus = 0.0
            if m_speaker and a_speaker and m_speaker == a_speaker:
                bonus += speaker_bonus
            if m_location and a_location and (
                m_location in a_location or a_location in m_location
            ):
                bonus += location_bonus
            if m_time and a_time and (
                m_time in a_time or a_time in m_time
            ):
                bonus += time_bonus

            sim_matrix[i, j] += bonus


# ---------------------------------------------------------------------------
# DP matching: weighted LCS with monotonicity constraint
# ---------------------------------------------------------------------------

def _dp_monotone_match(sim_matrix, threshold):
    """
    Find the highest-scoring monotone matching using dynamic programming.

    The alignment preserves order: if manga line i1 < i2 are both matched,
    their anime matches j1 < j2 must also be in order.

    Args:
        sim_matrix: (M, N) similarity matrix
        threshold: minimum similarity to consider a match

    Returns:
        List of (manga_idx, anime_idx, score) tuples
    """
    M, N = sim_matrix.shape
    if M == 0 or N == 0:
        return []

    # dp[i][j] = best total score considering manga 0..i-1 and anime 0..j-1
    # We use i=0..M, j=0..N where 0 means "no lines considered yet"
    dp = np.full((M + 1, N + 1), -np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    # choice[i][j]: 0=skip_manga, 1=skip_anime, 2=match
    choice = np.zeros((M + 1, N + 1), dtype=np.int8)

    for i in range(M + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                continue

            best = -np.inf
            best_choice = 0

            # Option 1: skip manga line i (advance i, keep j)
            if i > 0 and dp[i - 1, j] > best:
                best = dp[i - 1, j]
                best_choice = 0

            # Option 2: skip anime line j (keep i, advance j)
            if j > 0 and dp[i, j - 1] > best:
                best = dp[i, j - 1]
                best_choice = 1

            # Option 3: match manga i with anime j
            if i > 0 and j > 0:
                score = sim_matrix[i - 1, j - 1]
                if score >= threshold and dp[i - 1, j - 1] + score > best:
                    best = dp[i - 1, j - 1] + score
                    best_choice = 2

            dp[i, j] = best
            choice[i, j] = best_choice

    # Backtrack to find matched pairs
    matches = []
    i, j = M, N
    while i > 0 or j > 0:
        c = choice[i, j]
        if c == 2:  # match
            matches.append((i - 1, j - 1, float(sim_matrix[i - 1, j - 1])))
            i -= 1
            j -= 1
        elif c == 0:  # skip manga
            i -= 1
        else:  # skip anime
            j -= 1

    matches.reverse()
    return matches


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def _analyze_gaps(anchors, manga_dialogue, anime_dialogue):
    """
    Identify gaps between consecutive anchors and characterize them.

    Returns:
        List of gap dicts.
    """
    gaps = []

    # Build lookup: manga_seq -> dialogue entry, anime_seq -> dialogue entry
    manga_by_seq = {d["seq"]: d for d in manga_dialogue}
    anime_by_seq = {d["seq"]: d for d in anime_dialogue}

    manga_seqs = [d["seq"] for d in manga_dialogue]
    anime_seqs = [d["seq"] for d in anime_dialogue]

    # Add sentinel anchors at boundaries
    anchor_points = []
    anchor_points.append({"manga_seq": None, "anime_seq": None})  # before first
    for a in anchors:
        anchor_points.append(a)
    anchor_points.append({"manga_seq": None, "anime_seq": None})  # after last

    for gap_idx in range(len(anchor_points) - 1):
        start_anchor = anchor_points[gap_idx]
        end_anchor = anchor_points[gap_idx + 1]

        # Determine manga seq range in this gap
        if start_anchor["manga_seq"] is None:
            m_start = manga_seqs[0] if manga_seqs else None
        else:
            # Next manga seq after the start anchor
            start_pos = manga_seqs.index(start_anchor["manga_seq"])
            m_start = manga_seqs[start_pos + 1] if start_pos + 1 < len(manga_seqs) else None

        if end_anchor["manga_seq"] is None:
            m_end = manga_seqs[-1] if manga_seqs else None
        else:
            # Previous manga seq before the end anchor
            end_pos = manga_seqs.index(end_anchor["manga_seq"])
            m_end = manga_seqs[end_pos - 1] if end_pos > 0 else None

        # Determine anime seq range in this gap
        if start_anchor["anime_seq"] is None:
            a_start = anime_seqs[0] if anime_seqs else None
        else:
            start_pos = anime_seqs.index(start_anchor["anime_seq"])
            a_start = anime_seqs[start_pos + 1] if start_pos + 1 < len(anime_seqs) else None

        if end_anchor["anime_seq"] is None:
            a_end = anime_seqs[-1] if anime_seqs else None
        else:
            end_pos = anime_seqs.index(end_anchor["anime_seq"])
            a_end = anime_seqs[end_pos - 1] if end_pos > 0 else None

        # Collect unanchored seqs in this gap
        anchored_manga = {a["manga_seq"] for a in anchors}
        anchored_anime = {a["anime_seq"] for a in anchors}

        manga_unanchored = []
        if m_start is not None and m_end is not None:
            for seq in manga_seqs:
                if m_start <= seq <= m_end and seq not in anchored_manga:
                    manga_unanchored.append(seq)

        anime_unanchored = []
        if a_start is not None and a_end is not None:
            for seq in anime_seqs:
                if a_start <= seq <= a_end and seq not in anchored_anime:
                    anime_unanchored.append(seq)

        # Skip empty gaps
        if not manga_unanchored and not anime_unanchored:
            continue

        # Collect panel_ids and shot_ids from unanchored dialogue
        manga_panel_ids = list(dict.fromkeys(
            manga_by_seq[s].get("panel_id", "") for s in manga_unanchored
            if s in manga_by_seq and manga_by_seq[s].get("panel_id")
        ))
        anime_shot_ids = list(dict.fromkeys(
            anime_by_seq[s].get("shot_id") for s in anime_unanchored
            if s in anime_by_seq and anime_by_seq[s].get("shot_id") is not None
        ))

        # Anime time range
        anime_times = [
            (anime_by_seq[s].get("start_time_seconds", 0),
             anime_by_seq[s].get("end_time_seconds", 0))
            for s in anime_unanchored if s in anime_by_seq
        ]
        if anime_times:
            time_range = [
                min(t[0] for t in anime_times),
                max(t[1] for t in anime_times),
            ]
        else:
            time_range = [0.0, 0.0]

        gap = {
            "gap_id": len(gaps),
            "start_anchor_id": start_anchor.get("anchor_id"),
            "end_anchor_id": end_anchor.get("anchor_id"),
            "manga_seqs_unanchored": manga_unanchored,
            "anime_seqs_unanchored": anime_unanchored,
            "manga_panel_ids": manga_panel_ids,
            "anime_shot_ids": anime_shot_ids,
            "anime_time_range": time_range,
        }
        gaps.append(gap)

    return gaps


# ---------------------------------------------------------------------------
# Anime-original flagging
# ---------------------------------------------------------------------------

def _flag_anime_original(anchors, anime_dialogue, gaps):
    """
    Identify anime dialogue lines that are likely anime-original.

    Returns:
        List of anime seq numbers flagged as anime-original.
    """
    anchored_anime_seqs = {a["anime_seq"] for a in anchors}
    anime_original = []

    for gap in gaps:
        manga_count = len(gap["manga_seqs_unanchored"])
        anime_count = len(gap["anime_seqs_unanchored"])

        # If no manga panels in the gap but anime content exists -> anime-original
        if manga_count == 0 and anime_count > 0:
            anime_original.extend(gap["anime_seqs_unanchored"])
        # If ratio is very high -> likely anime-original padding
        elif manga_count > 0 and anime_count > manga_count * 10:
            # Flag excess anime lines beyond what's plausibly manga-derived
            excess = gap["anime_seqs_unanchored"][manga_count * 3:]
            anime_original.extend(excess)

    return anime_original


# ---------------------------------------------------------------------------
# Embedding comparison
# ---------------------------------------------------------------------------

def _compare_embeddings(matches_openai, matches_bow, manga_dialogue, anime_dialogue):
    """
    Compare results from OpenAI and BOW embeddings.

    Returns:
        Dict with comparison statistics.
    """
    # Build match lookup: manga_idx -> (anime_idx, score)
    openai_map = {m: (a, s) for m, a, s in matches_openai} if matches_openai else {}
    bow_map = {m: (a, s) for m, a, s in matches_bow} if matches_bow else {}

    all_manga_idxs = set(openai_map.keys()) | set(bow_map.keys())

    agree = 0
    disagree = 0
    openai_only = 0
    bow_only = 0
    score_pairs = []  # (openai_score, bow_score) for shared matches

    for m_idx in all_manga_idxs:
        in_openai = m_idx in openai_map
        in_bow = m_idx in bow_map

        if in_openai and in_bow:
            if openai_map[m_idx][0] == bow_map[m_idx][0]:
                agree += 1
                score_pairs.append((openai_map[m_idx][1], bow_map[m_idx][1]))
            else:
                disagree += 1
        elif in_openai:
            openai_only += 1
        else:
            bow_only += 1

    # Pearson correlation
    correlation = None
    if len(score_pairs) >= 3:
        openai_scores = np.array([p[0] for p in score_pairs])
        bow_scores = np.array([p[1] for p in score_pairs])
        if np.std(openai_scores) > 0 and np.std(bow_scores) > 0:
            correlation = float(np.corrcoef(openai_scores, bow_scores)[0, 1])

    return {
        "total_unique_matches": len(all_manga_idxs),
        "agree": agree,
        "disagree": disagree,
        "openai_only": openai_only,
        "bow_only": bow_only,
        "agreement_rate": agree / max(len(all_manga_idxs), 1),
        "pearson_correlation": correlation,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pass1(manga_dialogue, anime_dialogue, config=None):
    """
    Match manga dialogue lines to anime subtitle lines using dual embeddings.

    Both OpenAI and bag-of-words methods are run. The primary matching uses
    OpenAI scores (falling back to BOW if OpenAI is unavailable). Both scores
    are stored on each anchor for comparison.

    Args:
        manga_dialogue: list of dialogue entries from chapter_manifest.dialogue_index
        anime_dialogue: list of dialogue entries from episode_manifest.dialogue_index
        config: optional config overrides

    Returns:
        anchors: list of anchor dicts
        gaps: list of gap dicts
        anime_original_seqs: list of anime seq numbers flagged as anime-original
        comparison: dict with embedding method agreement stats
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    threshold = cfg["match_threshold"]

    bow_only = cfg.get("bow_only", False)

    print(f"[Pass 1] Matching {len(manga_dialogue)} manga lines "
          f"against {len(anime_dialogue)} anime lines...")

    if not manga_dialogue or not anime_dialogue:
        print("[Pass 1] No dialogue to match.")
        return [], [], [d["seq"] for d in anime_dialogue], {}

    # Validate API key unless --bow-only
    api_key = cfg.get("openai_api_key", "")
    if not bow_only and not api_key:
        print("ERROR: No OPENAI_API_KEY set.")
        print("  Either set the OPENAI_API_KEY environment variable, or")
        print("  use --bow-only to run with bag-of-words matching only.")
        return None, None, None, None

    # Extract text content
    manga_texts = [d.get("content", "") for d in manga_dialogue]
    anime_texts = [d.get("content", "") for d in anime_dialogue]
    all_texts = manga_texts + anime_texts
    n_manga = len(manga_texts)

    # --- Method 1: OpenAI embeddings ---
    openai_sim = None
    matches_openai = None

    if bow_only:
        print("[Pass 1] --bow-only mode: skipping OpenAI embeddings")
    else:
        print("[Pass 1] Computing OpenAI embeddings...")
        openai_embeddings = _embed_openai(all_texts, cfg)

        if openai_embeddings is not None:
            manga_emb_openai = openai_embeddings[:n_manga]
            anime_emb_openai = openai_embeddings[n_manga:]
            openai_sim = _cosine_similarity_matrix(manga_emb_openai, anime_emb_openai)
            # Apply metadata bonuses to a copy for matching
            openai_sim_boosted = openai_sim.copy()
            _apply_metadata_bonuses(openai_sim_boosted, manga_dialogue, anime_dialogue, cfg)
            matches_openai = _dp_monotone_match(openai_sim_boosted, threshold)
            print(f"[Pass 1] OpenAI: {len(matches_openai)} anchors found")
            print(f"[Pass 1] {get_embedding_cost_summary()}")
        else:
            print("ERROR: OpenAI embedding API call failed.")
            print("  Use --bow-only to run with bag-of-words matching only.")
            return None, None, None, None

    # --- Method 2: Bag-of-words cosine ---
    print("[Pass 1] Computing bag-of-words embeddings...")
    bow_embeddings = _embed_bow(all_texts)
    manga_emb_bow = bow_embeddings[:n_manga]
    anime_emb_bow = bow_embeddings[n_manga:]
    bow_sim = _cosine_similarity_matrix(manga_emb_bow, anime_emb_bow)
    bow_sim_boosted = bow_sim.copy()
    _apply_metadata_bonuses(bow_sim_boosted, manga_dialogue, anime_dialogue, cfg)
    matches_bow = _dp_monotone_match(bow_sim_boosted, threshold)
    print(f"[Pass 1] BOW: {len(matches_bow)} anchors found")

    # --- Choose primary matches ---
    if matches_openai is not None:
        primary_matches = matches_openai
        primary_method = "openai"
    else:
        primary_matches = matches_bow
        primary_method = "bow"

    # --- Compare methods ---
    comparison = _compare_embeddings(matches_openai, matches_bow,
                                     manga_dialogue, anime_dialogue)
    comparison["primary_method"] = primary_method

    # Print comparison
    print(f"\n[Pass 1] === Embedding Comparison ===")
    print(f"  Primary method: {primary_method}")
    print(f"  Agree on same match: {comparison['agree']}")
    print(f"  Disagree (different partner): {comparison['disagree']}")
    print(f"  OpenAI only: {comparison['openai_only']}")
    print(f"  BOW only: {comparison['bow_only']}")
    if comparison["pearson_correlation"] is not None:
        print(f"  Score correlation: {comparison['pearson_correlation']:.3f}")
    print()

    # --- Build anchor dicts ---
    # Build BOW match lookup for secondary scores
    bow_match_map = {}
    for m_idx, a_idx, score in matches_bow:
        bow_match_map[m_idx] = (a_idx, score)

    openai_match_map = {}
    if matches_openai:
        for m_idx, a_idx, score in matches_openai:
            openai_match_map[m_idx] = (a_idx, score)

    anchors = []
    for match_idx, (m_idx, a_idx, primary_score) in enumerate(primary_matches):
        m_line = manga_dialogue[m_idx]
        a_line = anime_dialogue[a_idx]

        # Get scores from both methods
        if primary_method == "openai":
            score_openai = float(openai_sim[m_idx, a_idx]) if openai_sim is not None else None
            bow_info = bow_match_map.get(m_idx)
            score_bow = float(bow_sim[m_idx, a_idx]) if bow_sim is not None else None
        else:
            score_openai = None
            score_bow = float(bow_sim[m_idx, a_idx])

        # Metadata match flags
        m_speaker = m_line.get("speaker_id") or ""
        a_speaker = a_line.get("speaker_id") or ""
        speaker_match = bool(m_speaker and a_speaker and m_speaker == a_speaker)

        m_loc = (m_line.get("location") or "").lower().strip()
        a_loc = (a_line.get("location") or "").lower().strip()
        location_match = bool(
            m_loc and a_loc and (m_loc in a_loc or a_loc in m_loc)
        )

        m_time = (m_line.get("time_context") or "").lower().strip()
        a_time = (a_line.get("time_context") or "").lower().strip()
        time_context_match = bool(
            m_time and a_time and (m_time in a_time or a_time in m_time)
        )

        anchor = {
            "anchor_id": f"anchor_{match_idx:03d}",
            "manga_seq": m_line["seq"],
            "anime_seq": a_line["seq"],
            "manga_content": m_line.get("content", ""),
            "anime_content": a_line.get("content", ""),
            "manga_panel_id": m_line.get("panel_id"),
            "manga_page": m_line.get("page"),
            "anime_shot_id": a_line.get("shot_id"),
            "anime_scene_number": a_line.get("scene_number"),
            "anime_start_time": a_line.get("start_time_seconds"),
            "anime_end_time": a_line.get("end_time_seconds"),
            "score_openai": score_openai,
            "score_bow": score_bow,
            "speaker_match": speaker_match,
            "location_match": location_match,
            "time_context_match": time_context_match,
            "manga_speaker_id": m_speaker or None,
            "anime_speaker_id": a_speaker or None,
            "manga_location": m_line.get("location"),
            "anime_location": a_line.get("location"),
        }
        anchors.append(anchor)

    print(f"[Pass 1] {len(anchors)} dialogue anchors established "
          f"({len(anchors)}/{len(manga_dialogue)} manga lines anchored, "
          f"{len(anchors)*100/max(len(manga_dialogue),1):.0f}%)")

    # --- Gap analysis ---
    print("[Pass 1] Analyzing gaps between anchors...")
    gaps = _analyze_gaps(anchors, manga_dialogue, anime_dialogue)
    print(f"[Pass 1] {len(gaps)} gaps identified")

    # --- Anime-original flagging ---
    anime_original_seqs = _flag_anime_original(anchors, anime_dialogue, gaps)
    print(f"[Pass 1] {len(anime_original_seqs)} anime dialogue lines flagged "
          f"as potentially anime-original")

    # Print sample anchors
    print(f"\n[Pass 1] === Sample Anchors ===")
    for a in anchors[:5]:
        oai = f"{a['score_openai']:.3f}" if a['score_openai'] is not None else "N/A"
        bow = f"{a['score_bow']:.3f}" if a['score_bow'] is not None else "N/A"
        spk = "Y" if a["speaker_match"] else "N"
        print(f"  {a['anchor_id']}: manga seq {a['manga_seq']} <-> "
              f"anime seq {a['anime_seq']}  "
              f"openai={oai} bow={bow} speaker={spk}")
        print(f"    M: \"{a['manga_content'][:60]}...\"")
        print(f"    A: \"{a['anime_content'][:60]}...\"")

    return anchors, gaps, anime_original_seqs, comparison
