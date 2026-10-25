"""Education levels: which school type, stream and year a document is written for.

Dutch schools split pupils by school type, not only by age: a vwo pupil and a vmbo-bb pupil
of the same age do very different maths. So each document gets its level from
docs/manifest.json, and a question can be limited to the material for one level.

Years from both school phases share one axis, so a filter can say "this year and below":
basisschool groep 1-8 is grade -1..6, voortgezet onderwijs leerjaar 1-6 is grade 7..12.
"""

import json
from pathlib import Path

MANIFEST = "manifest.json"

# The last year of each school type ("groep" for po, "leerjaar" for the others).
LAST_YEAR = {"po": 8, "vmbo": 4, "havo": 5, "vwo": 6}
# Streams per school type: the vmbo learning paths, and the maths subject in havo/vwo upper years.
STREAMS = {"po": set(), "vmbo": {"bb", "kb", "gl", "tl"}, "havo": {"a", "b", "d"}, "vwo": {"a", "b", "c", "d"}}
# Manifest fields copied to every chunk as they are, so an answer can show where it came from.
TEXT_FIELDS = ("domein", "licentie", "bron_url")


class LevelError(ValueError):
    """A level that does not exist, like havo 6 or vmbo with wiskunde B."""


def grade(niveau, year):
    """The year on the shared axis: groep 3 -> 1, groep 8 -> 6, leerjaar 1 -> 7, vwo 6 -> 12."""
    if niveau not in LAST_YEAR:
        raise LevelError(f"Unknown niveau '{niveau}', expected one of {sorted(LAST_YEAR)}.")
    if not 1 <= year <= LAST_YEAR[niveau]:
        raise LevelError(f"{niveau} has years 1 to {LAST_YEAR[niveau]}, not {year}.")
    return year - 2 if niveau == "po" else year + 6


def _stream(niveau, stroom):
    stroom = stroom.lower()
    if stroom not in STREAMS[niveau]:
        raise LevelError(f"{niveau} has no stroom '{stroom}', expected one of {sorted(STREAMS[niveau]) or 'none'}.")
    return stroom


def _as_list(value):
    return value if isinstance(value, list) else [value]


def chunk_metadata(entry):
    """Flatten one manifest entry into ChromaDB metadata (which allows no lists).

    Every school type becomes its own flag (niveau_havo: True) and so does every stream,
    with stroom_all for material that fits all streams. grade_min is the first year the
    material is for, which is what the "this year and below" filter compares.
    """
    niveaus = _as_list(entry["niveau"])
    if "po" in niveaus and len(niveaus) > 1:
        raise LevelError("Basisschool and voortgezet onderwijs material need separate files.")
    years = _as_list(entry.get("leerjaar", 1))
    meta = {"grade_min": grade(niveaus[0], min(years))}
    for niveau in niveaus:
        grade(niveau, max(years))  # every year must exist in every school type
        meta[f"niveau_{niveau}"] = True
    streams = [stroom.lower() for stroom in _as_list(entry.get("stroom") or [])]
    for stroom in streams:
        # Every listed school type must have the stream: flags are not paired per school type,
        # so havo+vwo with wiskunde C would otherwise reach havo, which has no C.
        missing = [niveau for niveau in niveaus if stroom not in STREAMS[niveau]]
        if missing:
            raise LevelError(f"{'/'.join(missing)} has no stroom '{stroom}'; use a separate file.")
        meta[f"stroom_{stroom}"] = True
    if not streams:
        meta["stroom_all"] = True
    meta.update({field: str(entry[field]) for field in TEXT_FIELDS if field in entry})
    return meta


def load_manifest(folder):
    """{file name: chunk metadata} from <folder>/manifest.json, or {} when there is none.

    A wrong entry stops the indexing with the file name in the message, because material
    filed under the wrong level would reach the wrong class without anyone noticing.
    """
    path = Path(folder) / MANIFEST
    if not path.exists():
        return {}
    levels = {}
    for name, entry in json.loads(path.read_text(encoding="utf-8")).items():
        try:
            levels[name] = chunk_metadata(entry)
        except (LevelError, KeyError, TypeError) as error:
            raise LevelError(f"{MANIFEST}, entry '{name}': {error}") from error
    return levels


def where(niveau, leerjaar=None, stroom=None):
    """ChromaDB filter for one level: this school type, this year and below, this stream.

    Lower years stay in because a teacher may want to repeat earlier material; higher years
    are left out, so a class never gets methods it has not learned yet.
    Chunks without level metadata never match, so a level question only sees labelled material.
    """
    if niveau not in LAST_YEAR:
        raise LevelError(f"Unknown niveau '{niveau}', expected one of {sorted(LAST_YEAR)}.")
    conditions = [{f"niveau_{niveau}": True}]
    if leerjaar is not None:
        conditions.append({"grade_min": {"$lte": grade(niveau, leerjaar)}})
    if stroom:
        conditions.append({"$or": [{f"stroom_{_stream(niveau, stroom)}": True}, {"stroom_all": True}]})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}
