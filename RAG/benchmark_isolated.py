"""
Benchmark script to compare FTS5 vs Elasticsearch vs Original matching.

FIXED VERSION - Addresses:
1. Applies outlier detection to timing measurements
2. Uses percentage-based parity tolerance
3. Fixes division by zero in overlap calculation
4. Adds better documentation and error handling
5. Reports outliers in CSV exports
"""

import time
import json
import os
import platform
import math
import sqlite3
import logging
import csv
import gc
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime

from fts5_indexer import FTS5DrugIndexer, SearchMode as FTS5SearchMode

try:
    from elasticsearch_indexer import ElasticsearchDrugIndexer, SearchMode as ESSearchMode
    import elasticsearch

    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False

ES_AVAILABLE = False
# =============================================================================
# Constants
# =============================================================================

# Outlier detection threshold: values > (p95 * OUTLIER_MULTIPLIER) are removed
OUTLIER_MULTIPLIER = 10.0

# Progress reporting interval (report every N iterations)
PROGRESS_INTERVAL = 25

# Maximum acceptable failure rate before aborting (10%)
MAX_FAILURE_RATE = 0.1

# Default parity tolerance as percentage of result count
DEFAULT_PARITY_TOLERANCE_PERCENT = 0.05  # 5%

# Minimum absolute parity tolerance
MIN_PARITY_TOLERANCE = 10


# =============================================================================
# Configuration
# =============================================================================


class SearchSemantics(Enum):
    """Search semantics for consistent behavior across engines"""

    TOKEN = "token"  # Exact-ish token matching
    PREFIX = "prefix"  # Token prefix matching
    SUBSTRING = "substring"  # Substring scan (FTS5+Original only)


class ResultMode(Enum):
    """Result limiting mode"""

    TOP_K = "top_k"  # Return top K results
    FULL = "full"  # Return all matching results


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""

    # Search settings
    search_semantics: SearchSemantics = SearchSemantics.TOKEN
    result_mode: ResultMode = ResultMode.TOP_K
    top_k_limit: int = 100

    # Benchmark settings
    iterations: int = 100
    warmup_iterations: int = 10

    # Parity / comparability thresholds
    # NOTE: parity_tolerance_count is now used as absolute tolerance
    # Combined with percentage-based calculation for adaptive behavior
    parity_tolerance_count: int = MIN_PARITY_TOLERANCE
    parity_min_overlap_ratio: float = 0.8
    parity_sample_k: int = 20

    # Outlier detection
    enable_outlier_detection: bool = True
    outlier_threshold_multiplier: float = OUTLIER_MULTIPLIER

    # Output settings
    output_prefix: str = "benchmark_results"

    def get_limit(self) -> Optional[int]:
        """Get the result limit based on mode"""
        if self.result_mode == ResultMode.FULL:
            return None
        return self.top_k_limit

    def calculate_parity_tolerance(self, count: int) -> int:
        """
        Calculate adaptive parity tolerance based on result count.

        Uses the larger of:
        - Configured absolute tolerance (parity_tolerance_count)
        - Percentage-based tolerance (5% of count)

        This ensures small result sets use absolute tolerance while
        large result sets use percentage-based tolerance.
        """
        percentage_tolerance = int(count * DEFAULT_PARITY_TOLERANCE_PERCENT)
        return max(self.parity_tolerance_count, percentage_tolerance)


# =============================================================================
# Reporter
# =============================================================================


class BenchmarkReporter:
    """Centralized reporter for benchmark logging and output"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.log_buffer: List[str] = []

        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(message)s",
        )
        self.logger = logging.getLogger("benchmark")

    def section(self, title: str):
        msg = f"\n{'=' * 80}\n{title}\n{'=' * 80}"
        self._output(msg)

    def subsection(self, title: str):
        msg = f"\n{title}\n{'-' * 80}"
        self._output(msg)

    def info(self, message: str):
        self._output(message)

    def progress(self, current: int, total: int, prefix: str = "Progress"):
        if self.verbose and current % PROGRESS_INTERVAL == 0:
            self._output(f"  {prefix}: {current}/{total}")

    def result(self, label: str, value: Any, indent: int = 2):
        spaces = " " * indent
        self._output(f"{spaces}{label}: {value}")

    def error(self, message: str):
        self._output(f"ERROR: {message}")

    def warning(self, message: str):
        self._output(f"WARNING: {message}")

    def _output(self, message: str):
        self.log_buffer.append(message)
        if self.verbose:
            print(message)

    def get_log(self) -> str:
        return "\n".join(self.log_buffer)


# =============================================================================
# Statistics
# =============================================================================

PERCENTILE_METHOD = "linear_interpolation_(n-1)_p"
PERCENTILE_FORMULA = "k=(n-1)*p; f=floor(k); c=min(f+1,n-1); x_p=x_f*(c-k)+x_c*(k-f)"


def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile using linear interpolation on sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    if f == c:
        return sorted_data[f]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


@dataclass
class PercentileStats:
    """Percentile-based statistics for a set of measurements (seconds)."""

    count: int
    min: float
    max: float
    median: float  # p50
    p95: float
    p99: float
    mean: float

    @classmethod
    def from_data(cls, data: List[float]) -> "PercentileStats":
        if not data:
            return cls(0, 0, 0, 0, 0, 0, 0)
        return cls(
            count=len(data),
            min=min(data),
            max=max(data),
            median=percentile(data, 50),
            p95=percentile(data, 95),
            p99=percentile(data, 99),
            mean=sum(data) / len(data),
        )

    @classmethod
    def from_data_with_outlier_detection(cls, data: List[float], outlier_threshold_multiplier: float = OUTLIER_MULTIPLIER) -> Tuple["PercentileStats", int]:
        """
        Calculate stats with outlier detection.

        Outliers are defined as values > threshold_multiplier * p95.

        Args:
            data: List of measurements
            outlier_threshold_multiplier: Multiplier for p95 to determine outlier threshold

        Returns:
            Tuple of (stats, num_outliers_removed)
        """
        if not data:
            return cls(0, 0, 0, 0, 0, 0, 0), 0

        sorted_data = sorted(data)
        p95 = percentile(sorted_data, 95)

        # Remove values > threshold * p95
        threshold = p95 * outlier_threshold_multiplier
        filtered_data = [x for x in sorted_data if x <= threshold]
        num_removed = len(sorted_data) - len(filtered_data)

        if not filtered_data:
            # Keep all if filter removes everything
            filtered_data = sorted_data
            num_removed = 0

        return cls.from_data(filtered_data), num_removed


# =============================================================================
# Comparable outcomes + stable identifiers
# =============================================================================


@dataclass
class ComparableQueryOutcome:
    """Per-query per-method outcome, including comparability signals."""

    result_count: int
    result_ids_sample: List[str]
    timings_ms: List[float]
    valid_semantics: bool
    parity_ok: bool
    overlap_ratio: Optional[float]
    rank_overlap_ratio: Optional[float]
    outliers_removed: int = 0
    included_in_summary: bool = False
    notes: str = ""


_ID_FALLBACK_FIELDS = (
    "brand_name",
    "generic_name",
    "substance_name",
    "manufacturer_name",
    "active_ingredient",
)


def _flatten_openfda_list(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x)
    return str(value) if value else ""


def _stable_hash_from_entry(entry: Dict[str, Any]) -> str:
    """
    Build a deterministic fallback id from a restricted, documented field set.

    IMPORTANT: This is only a fallback when FDA identifiers are unavailable.
    """
    openfda = entry.get("openfda") or {}

    parts: List[str] = []
    # Restricted stable fields (lowercased) to avoid hashing huge payloads.
    for k in _ID_FALLBACK_FIELDS:
        parts.append(_flatten_openfda_list(openfda.get(k, "")))
    joined = "|".join(p.strip().lower() for p in parts if p)
    if not joined:
        # Last resort: stable JSON serialization (sorted keys) of the entry.
        try:
            joined = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            joined = str(entry)
    digest = hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"hash:{digest}"


def canonical_doc_id(entry: Dict[str, Any]) -> str:
    """
    Canonical document identifier used across all engines.

    Precedence (first non-empty wins):
      1) entry['id'] (openFDA record id)
      2) entry['openfda']['set_id'] (first value if list)
      3) entry['openfda']['spl_id'] (first value if list)
      4) deterministic fallback hash over a restricted openfda field set (_ID_FALLBACK_FIELDS)

    NOTE: We prefix types (id:/set_id:/spl_id:/hash:) to avoid accidental collisions.
    """
    if not isinstance(entry, dict):
        digest = hashlib.sha1(str(entry).encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"hash:{digest}"

    if entry.get("id"):
        return f"id:{str(entry['id'])}"

    openfda = entry.get("openfda") or {}

    set_id = openfda.get("set_id")
    if isinstance(set_id, list) and set_id:
        return f"set_id:{str(set_id[0])}"
    if isinstance(set_id, str) and set_id:
        return f"set_id:{set_id}"

    spl_id = openfda.get("spl_id")
    if isinstance(spl_id, list) and spl_id:
        return f"spl_id:{str(spl_id[0])}"
    if isinstance(spl_id, str) and spl_id:
        return f"spl_id:{spl_id}"

    return _stable_hash_from_entry(entry)


def extract_doc_ids(results: List[Dict[str, Any]]) -> List[str]:
    """Normalize canonical document identifiers from result objects."""
    ids: List[str] = []
    for entry in results or []:
        ids.append(canonical_doc_id(entry))
    return ids


def _overlap_ratio(a: List[str], b: List[str], k: int) -> Optional[float]:
    """
    Overlap ratio of the first k IDs between lists a and b.
    Defined as |intersection| / min(k, len(a), len(b)).

    Args:
        a: First list of document IDs
        b: Second list of document IDs
        k: Number of top results to compare

    Returns:
        Overlap ratio [0.0, 1.0], or None if both lists are empty
    """
    a_k = a[:k]
    b_k = b[:k]
    denom = min(k, len(a_k), len(b_k))
    if denom <= 0:
        return None
    inter = len(set(a_k).intersection(set(b_k)))
    return inter / denom


def _rank_weighted_overlap(a: List[str], b: List[str], k: int) -> Optional[float]:
    """
    Rank-weighted overlap using discounted weights 1/log2(rank+2) for shared items.
    Normalized by the maximum possible weight for the top-k length.
    """
    a_k = a[:k]
    b_k = b[:k]
    denom_len = min(k, len(a_k), len(b_k))
    if denom_len <= 0:
        return None
    a_pos = {doc_id: idx for idx, doc_id in enumerate(a_k)}
    b_pos = {doc_id: idx for idx, doc_id in enumerate(b_k)}
    inter = set(a_pos).intersection(b_pos)
    if not inter:
        return 0.0
    total = 0.0
    for doc_id in inter:
        rank = min(a_pos[doc_id], b_pos[doc_id])
        total += 1.0 / math.log2(rank + 2)
    max_total = sum(1.0 / math.log2(rank + 2) for rank in range(denom_len))
    return total / max_total


# =============================================================================
# Deterministic sampling for overlap (avoid relevance-rank divergence)
# =============================================================================


def _sample_ids_sorted_by_doc_id(
    method: str,
    query: str,
    config: BenchmarkConfig,
    fts5: FTS5DrugIndexer,
    es: Optional["ElasticsearchDrugIndexer"],
) -> List[str]:
    """
    Return a deterministic ID sample for overlap checks.

    Rationale:
      FTS5 and Elasticsearch have different relevance scoring, so "top K" by rank can diverge
      even when the underlying match set is identical. For comparability gating, we sample
      a small, deterministic slice ordered by doc_id (where available) rather than score.

    Notes:
      - Uses the same semantic query constraints as the main search (TOKEN/PREFIX/SUBSTRING).
      - For Elasticsearch, sorts by the indexed keyword field doc_id ascending.
      - For FTS5, orders by drugs_data.doc_id ascending.
      - For Original, orders by canonical_doc_id (string) ascending over the full match set.
    """
    k = int(getattr(config, "parity_sample_k", 20) or 20)

    # Helper to normalize stored doc_id strings (doc_id is openFDA "id" here).
    def _id_from_doc_id(doc_id: str) -> str:
        return f"id:{doc_id}" if doc_id else ""

    if method == "fts5":
        try:
            q = (query or "").lower().strip()
            if config.search_semantics == SearchSemantics.SUBSTRING:
                pattern = f"%{q}%"
                fts5.cursor.execute(
                    """SELECT doc_id FROM drugs_data
                       WHERE LOWER(brand_names) LIKE ?
                          OR LOWER(generic_names) LIKE ?
                          OR LOWER(substance_names) LIKE ?
                          OR LOWER(active_ingredient) LIKE ?
                       ORDER BY doc_id ASC
                       LIMIT ?""",
                    (pattern, pattern, pattern, pattern, k),
                )
                rows = [r[0] for r in fts5.cursor.fetchall()]
                ids = [_id_from_doc_id(d) for d in rows if d]
                return ids

            # TOKEN/PREFIX via MATCH
            if config.search_semantics == SearchSemantics.TOKEN:
                match_q = f"{{brand_names generic_names substance_names active_ingredient}}: {q}"
            else:
                match_q = f"{{brand_names generic_names substance_names active_ingredient}}: {q}*"
            fts5.cursor.execute(
                """SELECT d.doc_id
                   FROM drugs_fts
                   JOIN drugs_data d ON drugs_fts.rowid = d.id
                   WHERE drugs_fts MATCH ?
                   ORDER BY d.doc_id ASC
                   LIMIT ?""",
                (match_q, k),
            )
            rows = [r[0] for r in fts5.cursor.fetchall()]
            ids = [_id_from_doc_id(d) for d in rows if d]
            return ids
        except Exception:
            # Fall back to ranked IDs (best-effort)
            r = fts5.search_drug_name(query, limit=config.get_limit(), mode=FTS5SearchMode.TOKEN if config.search_semantics == SearchSemantics.TOKEN else FTS5SearchMode.PREFIX)
            return sorted(extract_doc_ids(r))[:k]

    if method == "elasticsearch":
        if not ES_AVAILABLE or not es or config.search_semantics == SearchSemantics.SUBSTRING:
            return []
        try:
            es_mode = ESSearchMode.TOKEN if config.search_semantics == SearchSemantics.TOKEN else ESSearchMode.PREFIX
            body = {
                "query": es._build_drug_name_query((query or "").lower().strip(), es_mode),  # type: ignore[attr-defined]
                "size": k,
                "sort": [{"doc_id": {"order": "asc"}}],
            }
            resp = es.es.search(index=es.index_name, body=body)  # type: ignore[attr-defined]
            hits = resp.get("hits", {}).get("hits", []) or []
            ids: List[str] = []
            for h in hits:
                src = (h or {}).get("_source") or {}
                did = src.get("doc_id") or ""
                if did:
                    ids.append(_id_from_doc_id(str(did)))
                else:
                    # fallback to canonical over stored full_data
                    fd = src.get("full_data") or {}
                    ids.append(canonical_doc_id(fd))
            return ids
        except Exception:
            # Fall back to ranked IDs (best-effort)
            r = es.search_drug_name(query, limit=config.get_limit(), mode=ESSearchMode.TOKEN if config.search_semantics == SearchSemantics.TOKEN else ESSearchMode.PREFIX)
            return sorted(extract_doc_ids(r))[:k]

    return []


# =============================================================================
# System Info
# =============================================================================


def get_system_info(es: Optional["ElasticsearchDrugIndexer"] = None, fts5: Optional[FTS5DrugIndexer] = None) -> Dict[str, Any]:
    """Collect system information for reproducibility."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version,
        "timestamp": datetime.now().isoformat(),
    }

    # Best-effort SQLite compile options (useful when comparing builds)
    try:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute("PRAGMA compile_options;")
        opts = [row[0] for row in cur.fetchall()]
        info["sqlite_compile_options_count"] = len(opts)
    except Exception:
        info["sqlite_compile_options_count"] = "unknown"
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Try to get memory info
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024**3), 2)
        info["available_ram_gb"] = round(mem.available / (1024**3), 2)
    except ImportError:
        info["total_ram_gb"] = "psutil not installed"
        info["available_ram_gb"] = "psutil not installed"

    if ES_AVAILABLE:
        info["elasticsearch_client_version"] = getattr(elasticsearch, "__version__", "unknown")
        if es:
            info["elasticsearch_server_version"] = getattr(es, "get_server_version", lambda: "unknown")()
        else:
            info["elasticsearch_server_version"] = "unknown"

    if fts5:
        try:
            info["fts5_tokenizer"] = fts5.get_tokenizer()
            verified, note = getattr(fts5, "get_tokenizer_verified", lambda: (False, "unverified"))()
            info["fts5_tokenizer_verified"] = bool(verified)
            info["fts5_tokenizer_verification_note"] = note
        except Exception:
            info["fts5_tokenizer"] = "unknown"

    return info


# =============================================================================
# Original matcher (token/prefix/substring)
# =============================================================================

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


class OriginalStringMatcher:
    """
    Baseline matcher with explicit semantics (TOKEN / PREFIX / SUBSTRING).

    NOTE: This matcher uses regex-based tokenization which differs from
    FTS5's unicode61 tokenizer, leading to result count discrepancies.
    This is expected behavior as this serves as a reference baseline.
    """

    def __init__(self, data_dir: str = "fda_data", max_files: Optional[int] = None):
        self.data_dir = data_dir
        self.entries: List[Dict[str, Any]] = []
        self.search_texts: List[str] = []
        self.tokens_by_doc: List[set[str]] = []
        self.relkeys = ["brand_name", "generic_name", "substance_name", "manufacturer_name"]
        self.max_files = max_files
        self._load_data()

    def _tokenize(self, text: str) -> set[str]:
        """
        Approximation of FTS5 unicode61 tokenization.

        NOTE: This is NOT identical to FTS5's unicode61 tokenizer.
        Differences in word boundary detection and special character
        handling can lead to different result counts.
        """
        normalized = unicodedata.normalize("NFKD", text)
        without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        lowered = without_marks.lower()
        return set(match.group(0) for match in _TOKEN_RE.finditer(lowered))

    def _load_data(self):
        """Load all FDA data into memory (and precompute tokens for TOKEN/PREFIX)."""
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith(".json")]
        json_files.sort()  # stable order

        if self.max_files:
            json_files = json_files[: self.max_files]

        for json_file in json_files:
            file_path = os.path.join(self.data_dir, json_file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "results" not in data:
                    continue
                for result in data["results"]:
                    if "openfda" not in result:
                        continue
                    names: List[str] = []
                    for key in self.relkeys:
                        if key in result["openfda"]:
                            names.append(_flatten_openfda_list(result["openfda"][key]))
                    search_text = " ".join(names).lower()
                    self.search_texts.append(search_text)
                    self.tokens_by_doc.append(self._tokenize(search_text))
                    self.entries.append(result)

    def get_doc_count(self) -> int:
        return len(self.entries)

    def search_drug_name(self, drug_name: str, limit: Optional[int] = None, semantics: SearchSemantics = SearchSemantics.TOKEN) -> List[Dict[str, Any]]:
        """
        Search using explicit semantics:
          - TOKEN: query token must exist in document tokens
          - PREFIX: any token startswith(query)
          - SUBSTRING: substring scan over concatenated text (legacy behavior)
        """
        q = drug_name.strip().lower()
        results: List[Dict[str, Any]] = []

        if not q:
            return results

        if semantics == SearchSemantics.SUBSTRING:
            for i, text in enumerate(self.search_texts):
                if q in text:
                    results.append(self.entries[i])
                    if limit is not None and len(results) >= limit:
                        break
            return results

        if semantics == SearchSemantics.TOKEN:
            # Tokenize the query using the same method as documents
            query_tokens = self._tokenize(q)
            if not query_tokens:
                return results
            # All query tokens must be present in the document (AND semantics)
            for i, doc_toks in enumerate(self.tokens_by_doc):
                if query_tokens.issubset(doc_toks):
                    results.append(self.entries[i])
                    if limit is not None and len(results) >= limit:
                        break
            return results

        if semantics == SearchSemantics.PREFIX:
            # Simple prefix: any token in the document starting with the query string
            # This matches FTS5's behavior with the * operator
            for i, doc_toks in enumerate(self.tokens_by_doc):
                if any(t.startswith(q) for t in doc_toks):
                    results.append(self.entries[i])
                    if limit is not None and len(results) >= limit:
                        break
            return results

        # default: TOKEN
        for i, toks in enumerate(self.tokens_by_doc):
            if q in toks:
                results.append(self.entries[i])
                if limit is not None and len(results) >= limit:
                    break
        return results


# =============================================================================
# Benchmark runner
# =============================================================================


@dataclass
class MethodResults:
    outcomes_by_query: Dict[str, ComparableQueryOutcome] = field(default_factory=dict)

    def add_outcome(self, query: str, outcome: ComparableQueryOutcome) -> None:
        self.outcomes_by_query[query] = outcome

    def all_timings_ms_valid(self) -> List[float]:
        times: List[float] = []
        for outcome in self.outcomes_by_query.values():
            if outcome.valid_semantics and outcome.parity_ok:
                times.extend(outcome.timings_ms)
        return times

    def timings_ms_for_queries(self, query_set: set[str]) -> List[float]:
        times: List[float] = []
        for query, outcome in self.outcomes_by_query.items():
            if query in query_set and outcome.valid_semantics and outcome.parity_ok:
                times.extend(outcome.timings_ms)
        return times

    def get_stats_valid(self) -> PercentileStats:
        secs = [t / 1000.0 for t in self.all_timings_ms_valid()]
        return PercentileStats.from_data(secs)

    def get_query_stats_valid(self, query: str) -> PercentileStats:
        outcome = self.outcomes_by_query.get(query)
        if not outcome or not (outcome.valid_semantics and outcome.parity_ok):
            return PercentileStats.from_data([])
        secs = [t / 1000.0 for t in outcome.timings_ms]
        return PercentileStats.from_data(secs)


def _semantics_supported(method: str, semantics: SearchSemantics) -> Tuple[bool, str]:
    """Check if a method supports the given search semantics."""
    if semantics == SearchSemantics.SUBSTRING and method == "elasticsearch":
        return False, "Elasticsearch has no substring-equivalent query in this benchmark"
    return True, ""


def _snapshot_ram(reporter: BenchmarkReporter, label: str, snapshots: Dict[str, Any]) -> None:
    """Capture process RSS and system-available RAM, store in *snapshots* dict.

    Called at every phase boundary so the JSON / CSV export contains a full
    memory timeline.  Falls back gracefully when psutil is absent.
    """
    try:
        import psutil

        proc_rss_mb = round(psutil.Process().memory_info().rss / (1024**2), 1)
        sys_avail_gb = round(psutil.virtual_memory().available / (1024**3), 3)
        snapshots[label] = {"proc_rss_mb": proc_rss_mb, "sys_avail_gb": sys_avail_gb}
        reporter.info(f"  [RAM] {label}: process RSS {proc_rss_mb} MB  |  " f"system available {sys_avail_gb} GB")
    except ImportError:
        snapshots[label] = None
        reporter.info(f"  [RAM] {label}: psutil not installed — skipped")


def _warmup_single_method(reporter: BenchmarkReporter, method: str, queries: List[str], iterations: int, fts5: Optional[FTS5DrugIndexer], es: Optional["ElasticsearchDrugIndexer"], original: Optional[OriginalStringMatcher], config: BenchmarkConfig) -> None:
    """Warm a single method across all queries.

    Replaces the old all-methods-at-once ``run_warmup``.  Called once per
    method phase so that each method's warmup runs while it is the *only*
    engine in memory.
    """
    limit = config.get_limit()
    failed = 0
    total = len(queries) * iterations

    fts5_mode = FTS5SearchMode.TOKEN if config.search_semantics != SearchSemantics.PREFIX else FTS5SearchMode.PREFIX
    es_mode = (ESSearchMode.TOKEN if config.search_semantics != SearchSemantics.PREFIX else ESSearchMode.PREFIX) if ES_AVAILABLE else None

    for query in queries:
        for _ in range(iterations):
            try:
                if method == "fts5" and fts5:
                    if config.search_semantics == SearchSemantics.SUBSTRING:
                        fts5.search_substring(query, limit=limit)
                    else:
                        fts5.search_drug_name(query, limit=limit, mode=fts5_mode)
                elif method == "elasticsearch" and es:
                    if config.search_semantics != SearchSemantics.SUBSTRING:
                        es.search_drug_name(query, limit=limit, mode=es_mode)
                elif method == "original" and original:
                    original.search_drug_name(query, limit=limit, semantics=config.search_semantics)
            except Exception as e:
                failed += 1
                if failed > total * MAX_FAILURE_RATE:
                    reporter.error(f"Warmup failure rate exceeded for {method}: {failed}/{total}")
                    raise RuntimeError(f"Too many {method} warmup failures — check system health")
                reporter.warning(f"  {method} warmup failed for '{query}': {e}")

    if failed > 0:
        reporter.warning(f"  {method} warmup: {failed}/{total} failures ({failed / total:.1%})")
    reporter.info(f"  {method} warmup complete ({len(queries)} queries × {iterations} iterations).")


def _full_result_count(method: str, query: str, config: BenchmarkConfig, fts5: FTS5DrugIndexer, es: Optional["ElasticsearchDrugIndexer"], original: OriginalStringMatcher) -> Tuple[int, str]:
    """Return (full_count, note) for parity probing.

    Parity checks MUST use full result counts (not TOP-K truncated sizes).
    """
    if method == "fts5":
        try:
            if config.search_semantics == SearchSemantics.SUBSTRING:
                # True substring uses LIKE scan over drugs_data.
                pattern = f"%{(query or '').lower().strip()}%"
                fts5.cursor.execute(
                    """SELECT COUNT(*) FROM drugs_data
                       WHERE LOWER(brand_names) LIKE ?
                          OR LOWER(generic_names) LIKE ?
                          OR LOWER(substance_names) LIKE ?
                          OR LOWER(active_ingredient) LIKE ?""",
                    (pattern, pattern, pattern, pattern),
                )
                return int(fts5.cursor.fetchone()[0]), "count_via_like"
            # TOKEN/PREFIX use MATCH with the same query formulation as search_drug_name.
            q = (query or "").lower().strip()
            if config.search_semantics == SearchSemantics.TOKEN:
                match_q = f"{{brand_names generic_names substance_names active_ingredient}}: {q}"
            else:
                match_q = f"{{brand_names generic_names substance_names active_ingredient}}: {q}*"
            fts5.cursor.execute(
                """SELECT COUNT(*)
                   FROM drugs_fts
                   JOIN drugs_data d ON drugs_fts.rowid = d.id
                   WHERE drugs_fts MATCH ?""",
                (match_q,),
            )
            return int(fts5.cursor.fetchone()[0]), "count_via_match"
        except Exception as e:
            return 0, f"count_failed:{e}"

    if method == "elasticsearch":
        if not ES_AVAILABLE:
            return 0, "elasticsearch_not_available"
        if not es:
            return 0, "elasticsearch_not_enabled"
        try:
            es_mode = ESSearchMode.TOKEN if config.search_semantics == SearchSemantics.TOKEN else ESSearchMode.PREFIX
            # Prefer indexer helper if present.
            if hasattr(es, "count_drug_name"):
                return int(es.count_drug_name(query, mode=es_mode)), "count_via_es_count_api"
            # Fallback: count via ES client directly (best-effort).
            body = {"query": {"multi_match": {"query": (query or "").lower().strip(), "fields": ["brand_names", "generic_names", "substance_names", "active_ingredient"], "operator": "and"}}}
            res = es.es.count(index=es.index_name, body=body)  # type: ignore[attr-defined]
            return int(res.get("count", 0)), "count_via_es_count_api_fallback"
        except Exception as e:
            return 0, f"count_failed:{e}"

    if method == "original":
        try:
            # Full scan in memory; used once per query for parity probing.
            r = original.search_drug_name(query, limit=None, semantics=config.search_semantics)
            return len(r), "count_via_full_scan"
        except Exception as e:
            return 0, f"count_failed:{e}"

    return 0, "unknown_method"


def _probe_results(method: str, query: str, config: BenchmarkConfig, fts5: FTS5DrugIndexer, es: Optional["ElasticsearchDrugIndexer"], original: OriginalStringMatcher) -> Tuple[List[Dict[str, Any]], bool, str, int, str]:
    """Run a single non-timed search to capture a TOP-K ID sample and a FULL result count for parity."""
    limit = config.get_limit()

    supported, note = _semantics_supported(method, config.search_semantics)
    if not supported:
        return [], False, note, 0, "count_skipped"

    full_count, count_note = _full_result_count(method, query, config, fts5, es, original)

    if method == "fts5":
        if config.search_semantics == SearchSemantics.SUBSTRING:
            return fts5.search_substring(query, limit=limit), True, "", full_count, count_note
        fts5_mode = FTS5SearchMode.TOKEN if config.search_semantics == SearchSemantics.TOKEN else FTS5SearchMode.PREFIX
        return fts5.search_drug_name(query, limit=limit, mode=fts5_mode), True, "", full_count, count_note

    if method == "elasticsearch":
        if not es:
            return [], False, "Elasticsearch not enabled", 0, "count_skipped"
        es_mode = ESSearchMode.TOKEN if config.search_semantics == SearchSemantics.TOKEN else ESSearchMode.PREFIX
        return es.search_drug_name(query, limit=limit, mode=es_mode), True, "", full_count, count_note

    if method == "original":
        return original.search_drug_name(query, limit=limit, semantics=config.search_semantics), True, "", full_count, count_note

    return [], False, f"Unknown method: {method}", 0, "count_skipped"


def _time_search(method: str, query: str, config: BenchmarkConfig, fts5: FTS5DrugIndexer, es: Optional["ElasticsearchDrugIndexer"], original: OriginalStringMatcher, reporter: BenchmarkReporter) -> List[float]:
    """Run timed search iterations and return timings in milliseconds."""
    limit = config.get_limit()
    timings_ms: List[float] = []

    supported, _ = _semantics_supported(method, config.search_semantics)
    if not supported:
        return timings_ms

    # Pre-resolve modes
    fts5_mode = FTS5SearchMode.TOKEN if config.search_semantics == SearchSemantics.TOKEN else FTS5SearchMode.PREFIX
    es_mode = None
    if ES_AVAILABLE:
        es_mode = ESSearchMode.TOKEN if config.search_semantics == SearchSemantics.TOKEN else ESSearchMode.PREFIX

    failed_iterations = 0
    for i in range(config.iterations):
        try:
            start = time.perf_counter()
            if method == "fts5":
                if config.search_semantics == SearchSemantics.SUBSTRING:
                    fts5.search_substring(query, limit=limit)
                else:
                    fts5.search_drug_name(query, limit=limit, mode=fts5_mode)
            elif method == "elasticsearch":
                if es:
                    es.search_drug_name(query, limit=limit, mode=es_mode)
            elif method == "original":
                original.search_drug_name(query, limit=limit, semantics=config.search_semantics)
            else:
                break
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)
        except Exception as e:
            failed_iterations += 1
            if failed_iterations > config.iterations * MAX_FAILURE_RATE:
                reporter.error(f"Too many failures for {method}/{query}: {failed_iterations}")
                raise
            # Log but continue for transient errors
            reporter.warning(f"Search failed (iteration {i}): {e}")

        reporter.progress(i + 1, config.iterations)

    if failed_iterations > 0:
        reporter.warning(f"{method}/{query}: {failed_iterations} failed iterations out of {config.iterations}")

    return timings_ms


def run_benchmark(reporter: BenchmarkReporter, config: BenchmarkConfig, test_queries: List[str]) -> Tuple[Dict[str, MethodResults], Dict[str, Any]]:
    """Run the benchmark with **per-method isolation**.

    Architecture
    -------------
    The loop order is *method-first, queries-second*.  Each method runs in
    its own phase:

        snapshot_ram  →  init  →  snapshot_ram  →  warmup  →  snapshot_ram
        →  (probe + time) × N queries
        →  teardown  →  del  →  gc.collect()  →  snapshot_ram

    This guarantees that Original's in-process dataset (~2 GB of Python
    dicts) does **not** compete with FTS5's SQLite page cache or
    Elasticsearch's JVM heap during their timing phases.

    Parity checks run as a single post-phase pass over the stored probe
    data — no search engine is required for that step.
    """
    results: Dict[str, MethodResults] = {
        "fts5": MethodResults(),
        "elasticsearch": MethodResults(),
        "original": MethodResults(),
    }
    engine_settings: Dict[str, Any] = {}
    init_times_ms: Dict[str, float] = {}  # cold-start metric per method
    ram_snapshots: Dict[str, Any] = {}  # full memory timeline
    all_probes: Dict[str, Dict[str, Any]] = {  # probe data for parity (filled per-phase)
        "fts5": {},
        "elasticsearch": {},
        "original": {},
    }

    methods_in_order = ["fts5", "elasticsearch", "original"]

    # ── baseline RAM before anything is loaded ──────────────────────
    _snapshot_ram(reporter, "baseline", ram_snapshots)

    # ══════════════════════════════════════════════════════════════════
    # METHOD-ISOLATED TIMING PHASES
    # ══════════════════════════════════════════════════════════════════
    for method in methods_in_order:
        reporter.section(f"PHASE: {method.upper()}")

        # Local engine handles — only one is non-None at a time.
        fts5: Optional[FTS5DrugIndexer] = None  # type: ignore[no-redef]
        es: Optional["ElasticsearchDrugIndexer"] = None  # type: ignore[no-redef]
        original: Optional[OriginalStringMatcher] = None  # type: ignore[no-redef]

        _snapshot_ram(reporter, f"{method}:before_init", ram_snapshots)

        # ── INIT ─────────────────────────────────────────────────────
        init_start = time.perf_counter()

        if method == "fts5":
            reporter.info("Initializing FTS5…")
            fts5 = FTS5DrugIndexer(tokenizer="unicode61")
            fts5.connect()
            fts5_count = fts5.get_doc_count()
            tok = getattr(fts5, "get_tokenizer", lambda: "unknown")()
            tok_v, tok_n = getattr(fts5, "get_tokenizer_verified", lambda: (False, "unverified"))()
            engine_settings["fts5"] = {
                "tokenizer": tok,
                "tokenizer_verified": bool(tok_v),
                "tokenizer_verification_note": tok_n,
            }
            reporter.result("Documents indexed", fts5_count)

        elif method == "elasticsearch":
            if not ES_AVAILABLE:
                reporter.info("Elasticsearch not available " "(install: pip install elasticsearch).  Skipping phase.")
                continue  # ← skip rest of phase
            reporter.info("Initializing Elasticsearch…")
            try:
                es = ElasticsearchDrugIndexer()
                es.connect()
                is_valid, msg, es_count = es.validate_index()
                if not is_valid:
                    reporter.warning(msg)
                    reporter.info("Skipping Elasticsearch phase.")
                    es.close()
                    del es
                    gc.collect()
                    continue  # ← skip rest of phase
                reporter.result("Status", msg)
                engine_settings["elasticsearch"] = {
                    "server_version": getattr(es, "get_server_version", lambda: "unknown")(),
                    "analyzer_name": getattr(es, "get_analyzer_name", lambda: "unknown")(),
                    "index_name": getattr(es, "index_name", "unknown"),
                    "unbounded_size_cap": 10000,
                }
            except Exception as e:
                reporter.warning(f"Skipping Elasticsearch phase: {e}")
                if es:
                    try:
                        es.close()
                    except Exception:
                        pass
                    del es
                gc.collect()
                continue  # ← skip rest of phase

        elif method == "original":
            reporter.info("Initializing Original Matcher…")
            original = OriginalStringMatcher(max_files=None)
            original_count = original.get_doc_count()
            reporter.result("Documents loaded", original_count)

        # ── record init (cold-start) time ────────────────────────────
        init_ms = (time.perf_counter() - init_start) * 1000.0
        init_times_ms[method] = init_ms
        reporter.result("Init time", f"{init_ms:.1f} ms")

        _snapshot_ram(reporter, f"{method}:after_init", ram_snapshots)

        # ── WARMUP (single method only) ──────────────────────────────
        reporter.subsection(f"Warmup — {method}")
        _warmup_single_method(reporter, method, test_queries, config.warmup_iterations, fts5, es, original, config)

        _snapshot_ram(reporter, f"{method}:after_warmup", ram_snapshots)

        # ── PROBE + TIME ─────────────────────────────────────────────
        reporter.subsection(f"{method.upper()} — Queries " f"({config.iterations} iterations each)")
        reporter.info(f"Semantics: {config.search_semantics.value}  |  " f"Mode: {config.result_mode.value}" + (f" (limit={config.top_k_limit})" if config.result_mode == ResultMode.TOP_K else ""))

        for query in test_queries:
            reporter.subsection(f"{method} / '{query}'")

            # ── probe (non-timed): result count + doc-ID sample ──────
            (m_results, valid_sem, note, full_count, count_note) = _probe_results(method, query, config, fts5, es, original)
            doc_ids = extract_doc_ids(m_results)
            overlap_ids = _sample_ids_sorted_by_doc_id(method, query, config, fts5, es)

            all_probes[method][query] = {
                "valid_semantics": bool(valid_sem),
                "notes": note,
                "result_count_top_k": len(m_results),
                "full_result_count": int(full_count),
                "count_note": count_note,
                "result_ids": doc_ids,
                "result_ids_for_overlap": overlap_ids,
            }

            # ── timed iterations ─────────────────────────────────────
            timings_ms = _time_search(method, query, config, fts5, es, original, reporter)

            # immediate stats for log output
            if config.enable_outlier_detection and timings_ms:
                stats, outliers = PercentileStats.from_data_with_outlier_detection(
                    [t / 1000.0 for t in timings_ms],
                    outlier_threshold_multiplier=config.outlier_threshold_multiplier,
                )
            else:
                stats = PercentileStats.from_data([t / 1000.0 for t in timings_ms])
                outliers = 0

            reporter.info(f"  Median: {stats.median * 1000:.2f} ms  |  " f"P95: {stats.p95 * 1000:.2f} ms  |  " f"P99: {stats.p99 * 1000:.2f} ms  |  " f"Results: {full_count}  |  " f"valid_semantics={valid_sem}  |  outliers_removed={outliers}")

            # store outcome — parity fields are placeholders filled by
            # _compute_parity after all phases finish
            results[method].add_outcome(
                query,
                ComparableQueryOutcome(
                    result_count=full_count,
                    result_ids_sample=doc_ids[: config.parity_sample_k],
                    timings_ms=timings_ms,
                    valid_semantics=bool(valid_sem),
                    parity_ok=True,  # placeholder
                    overlap_ratio=None,  # placeholder
                    rank_overlap_ratio=None,  # placeholder
                    outliers_removed=outliers,
                    notes=note,
                ),
            )

        # ── TEARDOWN ─────────────────────────────────────────────────
        reporter.info(f"\n[Teardown] Closing {method} and releasing memory…")
        if fts5:
            fts5.close()
            del fts5
        if es:
            es.close()
            del es
        if original:
            del original

        gc.collect()  # return pages to OS
        _snapshot_ram(reporter, f"{method}:after_teardown", ram_snapshots)

    # ══════════════════════════════════════════════════════════════════
    # PARITY CHECKS  (post-timing; no engines required)
    # ══════════════════════════════════════════════════════════════════
    reporter.section("PARITY CHECKS  (post-timing; no engines required)")
    _compute_parity(reporter, config, results, all_probes, test_queries)

    # ── system info (engines already closed; details in engine_settings) ──
    system_info = get_system_info()
    system_info["ram_snapshots"] = ram_snapshots
    system_info["init_times_ms"] = init_times_ms

    return results, {"engine_settings": engine_settings, "system_info": system_info}


# =============================================================================
# Parity  (computed from stored probe data after all timing phases)
# =============================================================================


def _compute_parity(reporter: BenchmarkReporter, config: BenchmarkConfig, results: Dict[str, MethodResults], all_probes: Dict[str, Dict[str, Any]], test_queries: List[str]) -> None:
    """Fill parity fields on every stored ``ComparableQueryOutcome``.

    All overlap / count data was captured during the per-method probe step.
    This function needs no search engine — it is pure post-processing.

    Logic is identical to what the old ``run_benchmark`` did inline; it has
    been extracted so that it runs *after* every engine has been torn down.
    """
    methods_in_order = ["fts5", "elasticsearch", "original"]
    es_ran = bool(all_probes.get("elasticsearch"))  # did the ES phase produce data?

    for query in test_queries:
        reporter.subsection(f"Parity: '{query}'")

        # which methods actually produced probe data for this query?
        active = [m for m in methods_in_order if query in all_probes.get(m, {})]
        if not active:
            reporter.warning(f"  No probe data for '{query}' — skipping parity.")
            continue

        probe = {m: all_probes[m][query] for m in active}

        # ── pairwise overlap matrix ──────────────────────────────────
        overlap_matrix: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {}
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                m1, m2 = active[i], active[j]
                if not (probe[m1]["valid_semantics"] and probe[m2]["valid_semantics"]):
                    continue
                overlap = _overlap_ratio(
                    probe[m1]["result_ids_for_overlap"],
                    probe[m2]["result_ids_for_overlap"],
                    config.parity_sample_k,
                )
                rank_ov = _rank_weighted_overlap(
                    probe[m1]["result_ids_for_overlap"],
                    probe[m2]["result_ids_for_overlap"],
                    config.parity_sample_k,
                )
                overlap_matrix[(m1, m2)] = (overlap, rank_ov)

        # ── primary comparability gate ───────────────────────────────
        if es_ran and "fts5" in active and "elasticsearch" in active:
            ov_pair = overlap_matrix.get(("fts5", "elasticsearch"))
            primary_overlap = ov_pair[0] if ov_pair else None
            comparable_query = primary_overlap is not None and primary_overlap >= config.parity_min_overlap_ratio
        else:
            comparable_query = any(ovlp is not None and ovlp >= config.parity_min_overlap_ratio for ovlp, _ in overlap_matrix.values()) if overlap_matrix else True  # single method → trivially comparable

        # ── update each method's outcome ─────────────────────────────
        for method in active:
            outcome = results[method].outcomes_by_query.get(query)
            if not outcome:
                continue

            m_valid = probe[method]["valid_semantics"]
            m_count = probe[method]["full_result_count"]
            m_notes = outcome.notes  # may already have probe-time notes

            # gather all pairwise overlaps that involve this method
            overlaps_for_method: List[Tuple[str, Optional[float], Optional[float], int]] = []
            for (m1, m2), (ovlp, rank_ovlp) in overlap_matrix.items():
                if m1 == method:
                    overlaps_for_method.append((m2, ovlp, rank_ovlp, probe[m2]["full_result_count"]))
                elif m2 == method:
                    overlaps_for_method.append((m1, ovlp, rank_ovlp, probe[m1]["full_result_count"]))

            parity_ok = True
            overlap_val = None
            rank_overlap_val = None

            # ── baseline handling: Original is NOT gated when ES ran ──
            if method == "original" and es_ran and config.search_semantics != SearchSemantics.SUBSTRING:
                parity_ok = bool(m_valid)
                if overlaps_for_method:
                    vals = [o[1] for o in overlaps_for_method if o[1] is not None]
                    rvals = [o[2] for o in overlaps_for_method if o[2] is not None]
                    overlap_val = (sum(vals) / len(vals)) if vals else None
                    rank_overlap_val = (sum(rvals) / len(rvals)) if rvals else None
                m_notes = (m_notes + " | " if m_notes else "") + "baseline_not_gated_for_primary_comparison"

            elif not m_valid:
                parity_ok = False
                if not comparable_query:
                    m_notes = (m_notes + " | " if m_notes else "") + "non_comparable_query_no_majority_overlap"

            elif not comparable_query:
                parity_ok = False
                m_notes = (m_notes + " | " if m_notes else "") + "non_comparable_query_no_majority_overlap"

            elif overlaps_for_method:
                for other, ovlp, rank_ovlp, other_count in overlaps_for_method:
                    tolerance = config.calculate_parity_tolerance(max(m_count, other_count))
                    if abs(m_count - other_count) > tolerance:
                        parity_ok = False
                        m_notes = (m_notes + " | " if m_notes else "") + (f"Full result count mismatch vs {other} " f"(tolerance={tolerance}): {m_count} vs {other_count}")
                    if ovlp is not None and ovlp < config.parity_min_overlap_ratio:
                        parity_ok = False
                        m_notes = (m_notes + " | " if m_notes else "") + (f"Low top-id overlap vs {other}: " f"{ovlp:.3f} < {config.parity_min_overlap_ratio}")

                # store average overlaps for reporting
                vals = [o[1] for o in overlaps_for_method if o[1] is not None]
                rvals = [o[2] for o in overlaps_for_method if o[2] is not None]
                overlap_val = (sum(vals) / len(vals)) if vals else None
                rank_overlap_val = (sum(rvals) / len(rvals)) if rvals else None

            else:
                # only method that ran — trivially comparable
                parity_ok = bool(m_valid)
                overlap_val = 1.0 if parity_ok else None
                rank_overlap_val = 1.0 if parity_ok else None

            # ── apply outlier detection to stored timings ────────────
            outliers_removed = 0
            if config.enable_outlier_detection and outcome.timings_ms:
                _, outliers_removed = PercentileStats.from_data_with_outlier_detection(
                    [t / 1000.0 for t in outcome.timings_ms],
                    outlier_threshold_multiplier=config.outlier_threshold_multiplier,
                )

            # ── mutate outcome in place ──────────────────────────────
            outcome.parity_ok = parity_ok
            outcome.overlap_ratio = overlap_val
            outcome.rank_overlap_ratio = rank_overlap_val
            outcome.outliers_removed = outliers_removed
            outcome.notes = m_notes

            reporter.info(f"  {method}: count={m_count}  parity_ok={parity_ok}  " f"overlap={overlap_val}  rank_overlap={rank_overlap_val}")


# =============================================================================
# Summary + exports
# =============================================================================


def print_summary(reporter: BenchmarkReporter, results: Dict[str, MethodResults], config: BenchmarkConfig) -> None:
    """Print neutral summary table for valid comparable samples only."""
    reporter.section("SUMMARY TABLE (valid comparable samples only)")
    reporter.info("Includes timings only where valid_semantics=true and parity_ok=true for all enabled methods.\n")

    methods_in_order = ["fts5", "elasticsearch", "original"]

    # Primary comparison set: prefer FTS5 vs Elasticsearch when ES produced any valid outputs.
    es_enabled = any(outcome.valid_semantics for outcome in results["elasticsearch"].outcomes_by_query.values())
    primary_methods = ["fts5", "elasticsearch"] if es_enabled else ["fts5", "original"]
    enabled_methods = [method for method in methods_in_order if any(outcome.valid_semantics for outcome in results[method].outcomes_by_query.values())]
    if not enabled_methods:
        enabled_methods = methods_in_order

    valid_queries_per_method = {method: {query for query, outcome in results[method].outcomes_by_query.items() if outcome.valid_semantics and outcome.parity_ok} for method in primary_methods}
    common_valid_queries = set.intersection(*valid_queries_per_method.values()) if valid_queries_per_method else set()
    reporter.info(f"Common comparable queries: {len(common_valid_queries)}\n")

    rows: List[Tuple[str, int, float, float, float]] = []
    for method_name in methods_in_order:
        times = results[method_name].timings_ms_for_queries(common_valid_queries)
        stats = PercentileStats.from_data([t / 1000.0 for t in times])
        rows.append((method_name, stats.count, stats.median * 1000.0, stats.p95 * 1000.0, stats.p99 * 1000.0))

    # Print in a fixed-width table (no qualitative comparisons).
    reporter.info(f"{'method':<15} {'sample_size_valid':>16} {'median_ms':>12} {'p95_ms':>12} {'p99_ms':>12}")
    reporter.info("-" * 72)
    for method, n, median_ms, p95_ms, p99_ms in rows:
        reporter.info(f"{method:<15} {n:>16d} {median_ms:>12.2f} {p95_ms:>12.2f} {p99_ms:>12.2f}")

    # Quick note about non-comparable semantics mode
    if config.search_semantics == SearchSemantics.SUBSTRING:
        reporter.info("\nNote: SUBSTRING mode excludes Elasticsearch from comparability by design.")


def export_results(reporter: BenchmarkReporter, results: Dict[str, MethodResults], config: BenchmarkConfig, test_queries: List[str], system_info: Dict[str, Any], engine_settings: Dict[str, Any]) -> None:
    """Export results to JSON and CSV files (with validity metadata)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_scope = "client_query_only"

    methods_in_order = ["fts5", "elasticsearch", "original"]

    # Primary comparison set: prefer FTS5 vs Elasticsearch when ES produced any valid outputs.
    es_enabled = any(outcome.valid_semantics for outcome in results["elasticsearch"].outcomes_by_query.values())
    primary_methods = ["fts5", "elasticsearch"] if es_enabled else ["fts5", "original"]
    enabled_methods = [method for method in methods_in_order if any(outcome.valid_semantics for outcome in results[method].outcomes_by_query.values())]
    if not enabled_methods:
        enabled_methods = methods_in_order

    valid_queries_per_method = {method: {query for query, outcome in results[method].outcomes_by_query.items() if outcome.valid_semantics and outcome.parity_ok} for method in primary_methods}
    common_valid_queries = set.intersection(*valid_queries_per_method.values()) if valid_queries_per_method else set()
    common_query_list = [query for query in test_queries if query in common_valid_queries]
    excluded_query_count_by_method = {method: len(test_queries) - len(valid_queries_per_method.get(method, set())) for method in methods_in_order}

    # Helper: summary stats for valid comparable timings only
    summary_rows: List[Dict[str, Any]] = []
    for method in methods_in_order:
        times = results[method].timings_ms_for_queries(common_valid_queries)
        stats = PercentileStats.from_data([t / 1000.0 for t in times])
        summary_rows.append(
            {
                "method": method,
                "sample_size_valid": stats.count,
                "median_ms_valid": round(stats.median * 1000.0, 4),
                "p95_ms_valid": round(stats.p95 * 1000.0, 4),
                "p99_ms_valid": round(stats.p99 * 1000.0, 4),
                "common_query_count": len(common_valid_queries),
                "timing_scope": timing_scope,
            }
        )

    # Export JSON
    json_file = f"{config.output_prefix}_{timestamp}.json"
    export_data: Dict[str, Any] = {
        "timestamp": timestamp,
        "system_info": system_info,
        "config": {
            "search_semantics": config.search_semantics.value,
            "result_mode": config.result_mode.value,
            "top_k_limit": config.top_k_limit,
            "iterations": config.iterations,
            "warmup_iterations": config.warmup_iterations,
            "parity_tolerance_count": config.parity_tolerance_count,
            "parity_min_overlap_ratio": config.parity_min_overlap_ratio,
            "parity_sample_k": config.parity_sample_k,
            "enable_outlier_detection": config.enable_outlier_detection,
            "outlier_threshold_multiplier": config.outlier_threshold_multiplier,
        },
        "engine_settings": engine_settings,
        "test_queries": test_queries,
        "summary_valid": summary_rows,
        "common_query_count": len(common_valid_queries),
        "common_query_list": common_query_list,
        "excluded_query_count_by_method": excluded_query_count_by_method,
        "timing_scope": timing_scope,
        "percentile_method": PERCENTILE_METHOD,
        "percentile_formula": PERCENTILE_FORMULA,
        "per_query": {},
    }

    for query in test_queries:
        export_data["per_query"][query] = {}
        for method in ["fts5", "elasticsearch", "original"]:
            outcome = results[method].outcomes_by_query.get(query)
            if not outcome:
                continue
            included = bool(query in common_valid_queries and outcome.valid_semantics and outcome.parity_ok)
            export_data["per_query"][query][method] = {
                "valid_semantics": outcome.valid_semantics,
                "parity_ok": outcome.parity_ok,
                "overlap_ratio": outcome.overlap_ratio,
                "rank_overlap_ratio": outcome.rank_overlap_ratio,
                "result_count": outcome.result_count,
                "result_ids_sample": outcome.result_ids_sample,
                "timings_ms": [round(x, 4) for x in outcome.timings_ms],
                "outliers_removed": outcome.outliers_removed,
                "included_in_summary": included,
                "notes": outcome.notes,
            }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
    reporter.info(f"\nJSON results exported to: {json_file}")

    # Export summary CSV (standalone with config + engine notes)
    csv_file = f"{config.output_prefix}_{timestamp}.csv"
    summary_fields = [
        "method",
        "sample_size_valid",
        "median_ms_valid",
        "p95_ms_valid",
        "p99_ms_valid",
        "common_query_count",
        "timing_scope",
        "search_semantics",
        "result_mode",
        "top_k_limit",
        "iterations",
        "warmup_iterations",
        "parity_tolerance_count",
        "parity_min_overlap_ratio",
        "parity_sample_k",
        "enable_outlier_detection",
        "outlier_threshold_multiplier",
        "fts5_tokenizer",
        "es_analyzer_name",
        "es_server_version",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            out = dict(row)
            out.update(
                {
                    "search_semantics": config.search_semantics.value,
                    "result_mode": config.result_mode.value,
                    "top_k_limit": config.top_k_limit,
                    "iterations": config.iterations,
                    "warmup_iterations": config.warmup_iterations,
                    "parity_tolerance_count": config.parity_tolerance_count,
                    "parity_min_overlap_ratio": config.parity_min_overlap_ratio,
                    "parity_sample_k": config.parity_sample_k,
                    "enable_outlier_detection": config.enable_outlier_detection,
                    "outlier_threshold_multiplier": config.outlier_threshold_multiplier,
                    "common_query_count": len(common_valid_queries),
                    "timing_scope": timing_scope,
                    "fts5_tokenizer": (engine_settings.get("fts5") or {}).get("tokenizer", "unknown"),
                    "es_analyzer_name": (engine_settings.get("elasticsearch") or {}).get("analyzer_name", "unknown"),
                    "es_server_version": (engine_settings.get("elasticsearch") or {}).get("server_version", "unknown"),
                }
            )
            writer.writerow(out)
    reporter.info(f"Summary CSV exported to: {csv_file}")

    # Export per-query CSV - FIXED: Added outliers_removed field and reordered for readability
    per_query_csv = f"{config.output_prefix}_{timestamp}_per_query.csv"
    per_query_fields = [
        # Identity
        "query",
        "method",
        # Validity
        "valid_semantics",
        "parity_ok",
        "included_in_summary",
        # Results
        "result_count",
        "overlap_ratio",
        "rank_overlap_ratio",
        # Timings
        "sample_size",
        "outliers_removed",
        "min_ms",
        "median_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "mean_ms",
        # Config
        "search_semantics",
        "result_mode",
        "top_k_limit",
        "parity_tolerance_count",
        "parity_min_overlap_ratio",
        "parity_sample_k",
        "enable_outlier_detection",
        # Notes
        "notes",
    ]
    with open(per_query_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_query_fields)
        writer.writeheader()

        for query in test_queries:
            for method in ["fts5", "elasticsearch", "original"]:
                outcome = results[method].outcomes_by_query.get(query)
                if not outcome:
                    continue

                # FIXED: Apply outlier detection consistently
                if config.enable_outlier_detection and outcome.timings_ms:
                    stats, outliers = PercentileStats.from_data_with_outlier_detection([t / 1000.0 for t in outcome.timings_ms], outlier_threshold_multiplier=config.outlier_threshold_multiplier)
                else:
                    stats = PercentileStats.from_data([t / 1000.0 for t in outcome.timings_ms])
                    outliers = 0

                writer.writerow(
                    {
                        "query": query,
                        "method": method,
                        "valid_semantics": str(outcome.valid_semantics).lower(),
                        "parity_ok": str(outcome.parity_ok).lower(),
                        "included_in_summary": bool(query in common_valid_queries and outcome.valid_semantics and outcome.parity_ok),
                        "result_count": outcome.result_count,
                        "overlap_ratio": "" if outcome.overlap_ratio is None else f"{outcome.overlap_ratio:.4f}",
                        "rank_overlap_ratio": "" if outcome.rank_overlap_ratio is None else f"{outcome.rank_overlap_ratio:.4f}",
                        "sample_size": stats.count,
                        "outliers_removed": outliers,
                        "min_ms": f"{stats.min * 1000.0:.2f}",
                        "median_ms": f"{stats.median * 1000.0:.2f}",
                        "p95_ms": f"{stats.p95 * 1000.0:.2f}",
                        "p99_ms": f"{stats.p99 * 1000.0:.2f}",
                        "max_ms": f"{stats.max * 1000.0:.2f}",
                        "mean_ms": f"{stats.mean * 1000.0:.2f}",
                        "notes": outcome.notes,
                        "search_semantics": config.search_semantics.value,
                        "result_mode": config.result_mode.value,
                        "top_k_limit": config.top_k_limit,
                        "parity_tolerance_count": config.parity_tolerance_count,
                        "parity_min_overlap_ratio": config.parity_min_overlap_ratio,
                        "parity_sample_k": config.parity_sample_k,
                        "enable_outlier_detection": config.enable_outlier_detection,
                    }
                )
    reporter.info(f"Per-query CSV exported to: {per_query_csv}")


def main():
    reporter = BenchmarkReporter(verbose=True)

    # FIXED: Configuration with improved defaults
    config = BenchmarkConfig(
        search_semantics=SearchSemantics.TOKEN,
        result_mode=ResultMode.TOP_K,
        top_k_limit=100,
        iterations=100,
        warmup_iterations=10,
        parity_tolerance_count=MIN_PARITY_TOLERANCE,  # Now uses adaptive tolerance
        parity_min_overlap_ratio=0.8,
        parity_sample_k=20,
        enable_outlier_detection=True,  # FIXED: Enabled by default
        outlier_threshold_multiplier=OUTLIER_MULTIPLIER,
    )

    test_queries = [
        "aspirin",
        "ibuprofen",
        "metformin",
        "lisinopril",
        "amoxicillin",
        "tablet",
        "capsule",
        "injection",
        "hydro",
        "acid",
        "atorvastatin",
        "omeprazole",
        "levothyroxine",
        "amlodipine",
    ]

    test_queries = ["coumarin"]

    reporter.section("BENCHMARK CONFIGURATION")
    reporter.result("Search semantics", config.search_semantics.value)
    reporter.result("Result mode", config.result_mode.value)
    reporter.result("Top-K limit", config.top_k_limit)
    reporter.result("Iterations per query", config.iterations)
    reporter.result("Warmup iterations", config.warmup_iterations)
    reporter.result("Total queries", len(test_queries))
    reporter.result("Parity tolerance (min absolute)", config.parity_tolerance_count)
    reporter.result("Parity tolerance (percentage)", f"{DEFAULT_PARITY_TOLERANCE_PERCENT*100:.0f}%")
    reporter.result("Parity min overlap ratio", config.parity_min_overlap_ratio)
    reporter.result("Parity sample K", config.parity_sample_k)
    reporter.result("Outlier detection", "enabled" if config.enable_outlier_detection else "disabled")
    reporter.result("Outlier threshold multiplier", config.outlier_threshold_multiplier)

    # Run benchmark
    results, meta = run_benchmark(reporter, config, test_queries)

    # System info is captured with engine metadata from run_benchmark
    reporter.section("SYSTEM INFORMATION")
    for key, value in (meta.get("system_info") or {}).items():
        reporter.result(key, value)

    # Print summary
    print_summary(reporter, results, config)

    # Export results
    export_results(
        reporter,
        results,
        config,
        test_queries,
        system_info=meta.get("system_info") or {},
        engine_settings=meta.get("engine_settings") or {},
    )

    reporter.section("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
