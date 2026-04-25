"""
static_lookups.py — in-memory indices over static reference datasets.

Populated from the TSV files fetched by `scripts/fetch_static_data.py`:
  - gnomAD v4.1 per-gene constraint (pLI, LOEUF, missense-z)
  - HGNC canonical symbol ↔ Ensembl gene ID ↔ HGNC ID
  - Reactome leaf-pathway membership keyed by Ensembl gene ID

Design principles:
  1. Lazy: nothing is loaded until a query is made.
  2. Optional: missing files degrade to None returns — callers must handle.
  3. Stable keys: queries accept HGNC symbols OR any known alias/prev symbol
     via the HGNC alias→canonical map.
  4. Cheap to reload in tests: `StaticLookups(static_dir=...)` + `reset_lookups()`.

MCP-server call sites consult this layer *before* hitting the network, so a
populated `data/static/` directory lets a full AMD/CAD run avoid thousands of
gene-constraint + Reactome HTTPS round-trips.
"""
from __future__ import annotations

import csv
import logging
import os
import threading
from pathlib import Path
from typing import Optional

_LOG = logging.getLogger(__name__)

_DEFAULT_STATIC_DIR = Path(__file__).resolve().parent.parent / "data" / "static"

_GNOMAD_FILE   = "gnomad_constraint.tsv"
_HGNC_FILE     = "hgnc_complete.tsv"
_REACTOME_FILE = "reactome_ensembl2pathways.tsv"

# Column aliases per dataset — tolerant of minor naming drift between releases.
_GNOMAD_COLS = {
    "symbol":     ("gene", "gene_symbol", "symbol"),
    "gene_id":    ("gene_id", "ensembl_gene_id"),
    "pli":        ("lof.pLI", "pLI", "lof_pli", "pli"),
    "loeuf":      ("lof.oe_ci.upper", "lof.oe_ci_upper", "oe_lof_upper", "loeuf"),
    "oe_lof":     ("lof.oe", "oe_lof"),
    "mis_z":      ("mis.z_score", "mis.z", "mis_z", "missense_z"),
    "obs_lof":    ("lof.obs", "obs_lof"),
    "exp_lof":    ("lof.exp", "exp_lof"),
    "mane_only":  ("mane_select", "canonical"),
}

_HGNC_COLS = {
    "hgnc_id":     ("hgnc_id",),
    "symbol":      ("symbol",),
    "ensembl":     ("ensembl_gene_id",),
    "alias":       ("alias_symbol",),
    "prev":        ("prev_symbol",),
}


def _first_present(row: dict, aliases: tuple[str, ...]) -> str | None:
    for k in aliases:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _to_float(v: object) -> float | None:
    if v is None or v == "" or v == "NA":
        return None
    try:
        f = float(v)
        return f if f == f else None  # NaN → None
    except (TypeError, ValueError):
        return None


class StaticLookups:
    """Per-process lazy-loading lookup tables over static reference datasets.

    Each dataset loads on first access and is cached in memory.  A missing
    file sets a per-dataset `*_loaded` flag to False and causes queries on
    that dataset to return None — callers should treat None as "fall back
    to the live API".
    """

    def __init__(self, static_dir: Path | None = None) -> None:
        self._dir = Path(static_dir) if static_dir else _DEFAULT_STATIC_DIR
        self._lock = threading.RLock()

        self._gnomad: dict[str, dict] | None = None
        self._gnomad_loaded = False
        self._gnomad_by_ensembl: dict[str, dict] = {}

        self._hgnc_symbol_to_ensembl: dict[str, str] = {}
        self._hgnc_ensembl_to_symbol: dict[str, str] = {}
        self._hgnc_alias_to_canonical: dict[str, str] = {}
        self._hgnc_loaded = False

        self._reactome_by_ensembl: dict[str, list[dict]] = {}
        self._reactome_loaded = False

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _ensure_gnomad(self) -> None:
        with self._lock:
            if self._gnomad_loaded:
                return
            self._gnomad_loaded = True
            path = self._dir / _GNOMAD_FILE
            if not path.exists():
                _LOG.info("gnomAD constraint file missing: %s", path)
                self._gnomad = {}
                return
            tbl: dict[str, dict] = {}
            tbl_by_eid: dict[str, dict] = {}
            try:
                with path.open("r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh, delimiter="\t")
                    for row in reader:
                        sym = _first_present(row, _GNOMAD_COLS["symbol"])
                        if not sym:
                            continue
                        # gnomAD v4.1 has multiple transcript rows per gene — prefer
                        # MANE Select when available, otherwise keep the first we see.
                        mane = _first_present(row, _GNOMAD_COLS["mane_only"])
                        if sym in tbl and mane not in ("true", "True", "1"):
                            continue
                        rec = {
                            "gene":       sym,
                            "gene_id":    _first_present(row, _GNOMAD_COLS["gene_id"]),
                            "pli":        _to_float(_first_present(row, _GNOMAD_COLS["pli"])),
                            "loeuf":      _to_float(_first_present(row, _GNOMAD_COLS["loeuf"])),
                            "oe_lof":     _to_float(_first_present(row, _GNOMAD_COLS["oe_lof"])),
                            "missense_z": _to_float(_first_present(row, _GNOMAD_COLS["mis_z"])),
                            "n_obs_lof":  _to_float(_first_present(row, _GNOMAD_COLS["obs_lof"])),
                            "n_exp_lof":  _to_float(_first_present(row, _GNOMAD_COLS["exp_lof"])),
                            "data_source": "gnomAD v4.1 constraint (static)",
                        }
                        tbl[sym.upper()] = rec
                        eid = rec["gene_id"]
                        if eid:
                            tbl_by_eid[eid.split(".")[0].upper()] = rec
            except Exception as exc:
                _LOG.warning("failed to load gnomAD constraint: %s", exc)
                self._gnomad = {}
                return
            self._gnomad = tbl
            self._gnomad_by_ensembl = tbl_by_eid
            _LOG.info("gnomAD constraint loaded: %d genes", len(tbl))

    def _ensure_hgnc(self) -> None:
        with self._lock:
            if self._hgnc_loaded:
                return
            self._hgnc_loaded = True
            path = self._dir / _HGNC_FILE
            if not path.exists():
                _LOG.info("HGNC mapping file missing: %s", path)
                return
            try:
                with path.open("r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh, delimiter="\t")
                    for row in reader:
                        sym = _first_present(row, _HGNC_COLS["symbol"])
                        ens = _first_present(row, _HGNC_COLS["ensembl"])
                        if not sym:
                            continue
                        sym_u = sym.upper()
                        if ens:
                            self._hgnc_symbol_to_ensembl[sym_u] = ens
                            self._hgnc_ensembl_to_symbol[ens.upper()] = sym
                        # Alias + prev symbols map back to canonical
                        for alias_field in ("alias", "prev"):
                            raw = _first_present(row, _HGNC_COLS[alias_field])
                            if not raw:
                                continue
                            # HGNC uses "|"-separated lists inside the column
                            for alt in raw.replace('"', "").split("|"):
                                alt = alt.strip()
                                if alt and alt.upper() != sym_u:
                                    self._hgnc_alias_to_canonical.setdefault(alt.upper(), sym)
            except Exception as exc:
                _LOG.warning("failed to load HGNC mapping: %s", exc)
                return
            _LOG.info(
                "HGNC mapping loaded: %d symbols, %d aliases",
                len(self._hgnc_symbol_to_ensembl), len(self._hgnc_alias_to_canonical),
            )

    def _ensure_reactome(self) -> None:
        with self._lock:
            if self._reactome_loaded:
                return
            self._reactome_loaded = True
            path = self._dir / _REACTOME_FILE
            if not path.exists():
                _LOG.info("Reactome mapping file missing: %s", path)
                return
            tbl: dict[str, list[dict]] = {}
            try:
                with path.open("r", encoding="utf-8", newline="") as fh:
                    # Ensembl2Reactome.txt has no header — 6 tab-separated columns.
                    reader = csv.reader(fh, delimiter="\t")
                    for row in reader:
                        if len(row) < 6:
                            continue
                        ensembl, path_id, _url, path_name, _evidence, species = row[:6]
                        if "Homo sapiens" not in species:
                            continue
                        base_eid = (ensembl or "").split(".")[0].upper()
                        if not base_eid:
                            continue
                        tbl.setdefault(base_eid, []).append({
                            "pathway_id": path_id,
                            "name":       path_name,
                            "species":    species,
                        })
            except Exception as exc:
                _LOG.warning("failed to load Reactome mapping: %s", exc)
                return
            self._reactome_by_ensembl = tbl
            _LOG.info("Reactome mapping loaded: %d genes", len(tbl))

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def _canonical_symbol(self, gene: str) -> str:
        """Resolve aliases and previous symbols to the current HGNC symbol.
        Falls through to the input when HGNC is not loaded or the input is
        already canonical.  Always upper-cases for lookup."""
        if not gene:
            return ""
        g_u = gene.upper()
        self._ensure_hgnc()
        if g_u in self._hgnc_symbol_to_ensembl:
            return g_u
        alt = self._hgnc_alias_to_canonical.get(g_u)
        return alt.upper() if alt else g_u

    def get_gnomad_constraint(self, gene: str) -> dict | None:
        """Return a per-gene gnomAD constraint dict or None if not available.
        Dict shape matches `mcp_servers.ukb_wes_server.get_gnomad_constraint`
        for drop-in replacement:
            {gene, gene_id, pli, loeuf, oe_lof, missense_z,
             n_obs_lof, n_exp_lof, data_source}
        """
        if not gene:
            return None
        self._ensure_gnomad()
        if not self._gnomad:
            return None
        # Try canonical symbol first, then aliases.
        canon = self._canonical_symbol(gene)
        rec = self._gnomad.get(canon) or self._gnomad.get(gene.upper())
        if rec is None:
            # Last-ditch: resolve symbol → Ensembl via HGNC then look up by gene_id.
            ens = self.get_ensembl_id(gene)
            if ens:
                rec = self._gnomad_by_ensembl.get(ens.split(".")[0].upper())
        return rec

    def get_pli(self, gene: str) -> float | None:
        rec = self.get_gnomad_constraint(gene)
        return rec.get("pli") if rec else None

    def get_loeuf(self, gene: str) -> float | None:
        rec = self.get_gnomad_constraint(gene)
        return rec.get("loeuf") if rec else None

    def get_ensembl_id(self, symbol: str) -> str | None:
        """Return the canonical Ensembl gene ID (unversioned) for a HGNC symbol
        or any of its known aliases.  None if HGNC not loaded or unknown."""
        if not symbol:
            return None
        canon = self._canonical_symbol(symbol)
        return self._hgnc_symbol_to_ensembl.get(canon)

    def get_symbol_from_ensembl(self, ensembl_id: str) -> str | None:
        if not ensembl_id:
            return None
        self._ensure_hgnc()
        key = ensembl_id.split(".")[0].upper()
        return self._hgnc_ensembl_to_symbol.get(key)

    def get_reactome_pathways(self, gene_or_ensembl: str) -> list[dict] | None:
        """Return Reactome leaf-pathway dicts for a gene symbol or Ensembl ID.
        None = mapping file not available; [] = no pathways for this gene."""
        if not gene_or_ensembl:
            return None
        self._ensure_reactome()
        if not self._reactome_by_ensembl:
            return None
        arg = gene_or_ensembl.strip()
        if arg.upper().startswith("ENSG"):
            key = arg.split(".")[0].upper()
        else:
            ens = self.get_ensembl_id(arg)
            if not ens:
                return []
            key = ens.split(".")[0].upper()
        return list(self._reactome_by_ensembl.get(key, []))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Report which datasets are loaded and how many rows they contain.
        Forces a load of each — use for diagnostics only, not on hot paths."""
        self._ensure_gnomad()
        self._ensure_hgnc()
        self._ensure_reactome()
        return {
            "static_dir":        str(self._dir),
            "gnomad_loaded":     bool(self._gnomad),
            "gnomad_n_genes":    len(self._gnomad or {}),
            "hgnc_loaded":       bool(self._hgnc_symbol_to_ensembl),
            "hgnc_n_symbols":    len(self._hgnc_symbol_to_ensembl),
            "hgnc_n_aliases":    len(self._hgnc_alias_to_canonical),
            "reactome_loaded":   bool(self._reactome_by_ensembl),
            "reactome_n_genes":  len(self._reactome_by_ensembl),
        }


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_singleton: Optional[StaticLookups] = None
_singleton_lock = threading.Lock()


def get_lookups(static_dir: Path | None = None) -> StaticLookups:
    """Return the process-wide StaticLookups.  Passing `static_dir` or setting
    the STATIC_DATA_DIR env var overrides the default `data/static/` location
    and rebuilds the singleton (useful for tests)."""
    global _singleton
    override = static_dir or os.getenv("STATIC_DATA_DIR")
    if override is not None:
        with _singleton_lock:
            _singleton = StaticLookups(Path(override))
            return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = StaticLookups()
        return _singleton


def reset_lookups() -> None:
    """Drop the singleton so the next get_lookups() rebuilds.  Test-only."""
    global _singleton
    with _singleton_lock:
        _singleton = None
