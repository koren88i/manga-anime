"""
On-demand API client for manga chapter data from the One Piece Fandom Wiki.

Provides structured data for chapters (characters, summaries, notes) without
bulk scraping.  Each function makes one or two HTTP GET requests via the
shared MediaWiki helpers in one_piece_wiki_api.py.

Usage:
    from manga_wiki_api import get_chapter_characters, get_chapter_summary, ...

    chars = get_chapter_characters(1)
    # [{"name": "Shanks", "wiki_page": "Shanks", "annotation": None, "group": "Red Hair Pirates"}, ...]

    summary = get_chapter_summary(1)
    # "The chapter opens with..."

    info = get_chapter_infobox(1)
    # {"chapter": 1, "volume": "1", "title": "Romance Dawn ...", ...}
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import re

from wiki.one_piece_wiki_api import (
    _api_get,
    _get_section_wikitext,
    _get_section_index,
    _get_rendered_html,
    _strip_wikitext_markup,
    _InfoboxParser,
    get_story_arcs,
    get_character_info,
    get_character_image,
)


# ---------------------------------------------------------------------------
# Public API -- Chapters
# ---------------------------------------------------------------------------

def get_chapter_characters(chapter_num):
    """
    Get the list of characters appearing in a chapter.

    Parses the CharTable wikitext format used on chapter pages, which groups
    characters by faction/location (e.g. "Red Hair Pirates", "Foosha Village").

    Returns:
        list of dicts: [{"name": str, "wiki_page": str,
                         "annotation": str|None, "group": str|None}, ...]
    """
    page = f"Chapter_{chapter_num}"
    idx = _get_section_index(page, "Characters")
    if idx is None:
        return []

    wikitext = _get_section_wikitext(page, idx)
    characters = []
    current_group = None

    for line in wikitext.split("\n"):
        line = line.strip()

        # Skip table markers, header rows, row separators, empty cells
        if not line or line.startswith("{|") or line.startswith("|}"):
            continue
        if line.startswith("!") or line.startswith("|-"):
            continue
        if line == "|":
            continue

        # Group header: ;[[Red Hair Pirates]] or ;Foosha Village
        if line.startswith(";"):
            group_text = line.lstrip(";").strip()
            current_group = _strip_wikitext_markup(group_text)
            continue

        # Character entry: *[[Page|Display]] ''(annotation)''
        if line.startswith("*"):
            line = line.lstrip("*").strip()

            # Extract annotation like ''(flashback)'' or ''(cover)''
            annotation = None
            ann_match = re.search(r"''\(([^)]+)\)''", line)
            if ann_match:
                annotation = ann_match.group(1)
                line = line[:ann_match.start()].strip()

            # Extract wiki link [[Page|Display]] or [[Page]]
            link_match = re.search(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]", line)
            if link_match:
                wiki_page = link_match.group(1).strip()
                display = (link_match.group(2) or wiki_page).strip()
                characters.append({
                    "name": display,
                    "wiki_page": wiki_page,
                    "annotation": annotation,
                    "group": current_group,
                })

    return characters


def get_chapter_summary(chapter_num):
    """
    Get the long summary for a chapter.

    Returns:
        str: Plain-text chapter summary.
    """
    page = f"Chapter_{chapter_num}"
    idx = _get_section_index(page, "Long Summary")
    if idx is None:
        return ""

    wikitext = _get_section_wikitext(page, idx)
    lines = wikitext.split("\n")
    lines = [l for l in lines if not l.startswith("==")]
    text = "\n".join(lines)
    return _strip_wikitext_markup(text)


def get_chapter_short_summary(chapter_num):
    """
    Get the short summary for a chapter.

    Returns:
        str: Plain-text short summary.
    """
    page = f"Chapter_{chapter_num}"
    idx = _get_section_index(page, "Short Summary")
    if idx is None:
        return ""

    wikitext = _get_section_wikitext(page, idx)
    lines = wikitext.split("\n")
    lines = [l for l in lines if not l.startswith("==")]
    text = "\n".join(lines)
    return _strip_wikitext_markup(text)


def get_chapter_infobox(chapter_num):
    """
    Get chapter metadata from the rendered portable infobox.

    Returns:
        dict with keys like: chapter, volume, title, viz_title, pages,
        release_date, wsj_issue, anime, etc.
    """
    page = f"Chapter_{chapter_num}"
    html = _get_rendered_html(page, section=0)
    parser = _InfoboxParser()
    parser.feed(html)

    result = {"chapter": chapter_num}

    # Filter out navigation/chronology noise from the infobox
    skip_keys = {"chapter_chronology", "previous", "next", "next_"}
    for label, value in parser.get_clean_fields().items():
        key = label.lower().replace(" ", "_").replace(":", "")
        if any(sk in key for sk in skip_keys):
            continue
        # Skip values that are just nav arrows
        if value.strip() in ("Previous", "Next", ""):
            continue
        result[key] = value

    return result


def get_chapter_notes(chapter_num):
    """
    Get the Chapter Notes section (key plot points and revelations).

    Returns:
        list of str: Each bullet point as plain text.
    """
    page = f"Chapter_{chapter_num}"
    idx = _get_section_index(page, "Chapter Notes")
    if idx is None:
        return []

    wikitext = _get_section_wikitext(page, idx)
    notes = []
    for line in wikitext.split("\n"):
        line = line.strip()
        if line.startswith("*"):
            text = _strip_wikitext_markup(line.lstrip("*").strip())
            if text:
                notes.append(text)
    return notes


def get_chapter_cover_page(chapter_num):
    """
    Get the cover page description.

    Returns:
        str: Plain-text cover page description.
    """
    page = f"Chapter_{chapter_num}"
    idx = _get_section_index(page, "Cover Page")
    if idx is None:
        return ""

    wikitext = _get_section_wikitext(page, idx)
    lines = wikitext.split("\n")
    lines = [l for l in lines if not l.startswith("==")]
    text = "\n".join(lines)
    return _strip_wikitext_markup(text)


def get_arc_for_chapter(chapter_num):
    """
    Look up which arc and saga a chapter belongs to.

    Args:
        chapter_num: Chapter number (int)

    Returns:
        dict: {"saga": str, "arc": str, "wiki_page": str,
               "chapters": "X-Y", "episodes": "X-Y"} or None
    """
    sagas = get_story_arcs()
    for saga in sagas:
        for arc in saga["arcs"]:
            ch_str = arc.get("chapters", "")
            if not ch_str:
                continue
            parts = ch_str.split("-")
            if len(parts) == 2:
                try:
                    start, end = int(parts[0]), int(parts[1])
                    if start <= chapter_num <= end:
                        return {
                            "saga": saga["saga"],
                            "arc": arc["name"],
                            "wiki_page": arc["wiki_page"],
                            "chapters": ch_str,
                            "episodes": arc.get("episodes", ""),
                        }
                except ValueError:
                    continue
    return None


# ---------------------------------------------------------------------------
# Convenience: get everything for a chapter in one call
# ---------------------------------------------------------------------------

def get_chapter_data(chapter_num):
    """
    Fetch all available wiki data for a chapter.

    Returns:
        dict with keys: characters, summary, short_summary, infobox,
        chapter_notes, cover_page
    """
    return {
        "chapter": chapter_num,
        "characters": get_chapter_characters(chapter_num),
        "summary": get_chapter_summary(chapter_num),
        "short_summary": get_chapter_short_summary(chapter_num),
        "infobox": get_chapter_infobox(chapter_num),
        "chapter_notes": get_chapter_notes(chapter_num),
        "cover_page": get_chapter_cover_page(chapter_num),
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io

    # Force UTF-8 output on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

    chapter = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    print(f"=== Chapter {chapter} Characters ===")
    for c in get_chapter_characters(chapter):
        ann = f" ({c['annotation']})" if c["annotation"] else ""
        grp = f" [{c['group']}]" if c["group"] else ""
        print(f"  - {c['name']}{ann}{grp}  (wiki: {c['wiki_page']})")

    print(f"\n=== Chapter {chapter} Short Summary ===")
    print(get_chapter_short_summary(chapter))

    print(f"\n=== Chapter {chapter} Long Summary ===")
    summary = get_chapter_summary(chapter)
    print(summary[:500] + "..." if len(summary) > 500 else summary)

    print(f"\n=== Chapter {chapter} Infobox ===")
    infobox = get_chapter_infobox(chapter)
    for k, v in infobox.items():
        print(f"  {k}: {v}")

    print(f"\n=== Chapter {chapter} Notes ===")
    for note in get_chapter_notes(chapter):
        print(f"  * {note[:150]}{'...' if len(note) > 150 else ''}")

    print(f"\n=== Chapter {chapter} Cover Page ===")
    cover = get_chapter_cover_page(chapter)
    print(cover[:300] + "..." if len(cover) > 300 else cover)

    print(f"\n=== Arc for Chapter {chapter} ===")
    arc_lookup = get_arc_for_chapter(chapter)
    if arc_lookup:
        for k, v in arc_lookup.items():
            print(f"  {k}: {v}")
    else:
        print("  (not found)")
