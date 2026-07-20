"""
data/feature_registry.py

Feature registry for NonB, TE, and NonB2 (RNA secondary structure + rG4) feature blocks.

This is the source for:
  - Feature names, in the exact order they appear in the cleaned CSVs
  - Sub-group assignment (e.g. APR, GQ, TE_LINE, TE_LCTR, RG4, SS_STAB ...)
  - Statistical category (count, presence, abs_length, rel_length, density, gap, quality,
    fraction, stability, binned_prop)
  - Scaling strategy per category
  - Block source type: "both" (genomic + processed), "processed" (single source)

* The registry is constructed from canonical feature name lists, so adding or
  removing a feature only requires editing the relevant name lists below.
* Scaling is defined per (block, category) pair and applied by FeatureScalerBank.
  The registry itself does not scale anything.
* Block source type controls how SubgroupProjectionLayer receives its inputs:
    "both"      — cat([x_genomic, x_processed]) fed to each subgroup MLP
    "processed" — x_processed only (single tensor)
    "genomic"   — x_genomic only (single tensor)

Sub-groups
----------
NonB  : APR | DR | GQ | IR | MR | STR | TRI | Z | GLOBAL
TE    : TE_CORE | TE_LCTR | TE_PSEUDO | TE_UNKNOWN | TE_GLOBAL | TE_QUALITY
NonB2 : RG4 | SS_STAB | SS_COUNT | SS_RELPOS | SS_BINNED

Statistical categories
----------------------
presence      — binary flag (0/1)               → scale: none
count         — integer hit/type counts          → scale: none
abs_length    — absolute bp lengths              → scale: robust
rel_length    — percentage of transcript (pct)   → scale: minmax
density       — counts or lengths per kb         → scale: robust
gap           — inter-hit gap statistics         → scale: robust
quality       — alignment scores, divergence     → scale: robust (TE only)
diversity     — motif/family richness counts     → scale: none
fraction      — fraction/proportion [0, 1]       → scale: minmax
stability     — Z-score, MFE, ED statistics      → scale: robust
binned_prop   — per-bin proportion [0, 1]        → scale: minmax
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureMeta:
    name:     str   # exact column name in the CSV
    block:    str   # "nonb" | "te" | "nonb2"
    subgroup: str   # e.g. "APR", "TE_CORE", "RG4", "SS_STAB" ...
    category: str   # presence | count | abs_length | rel_length | density |
                    # gap | quality | diversity | fraction | stability | binned_prop
    scale:    str   # none | robust | minmax


@dataclass(frozen=True)
class BlockMeta:
    """Metadata for a feature block (nonb / te / nonb2)."""
    name:        str         # "nonb" | "te" | "nonb2"
    source:      str         # "both" | "processed" | "genomic"
    csv_key:     str         # config key for the CSV path(s)
                             # "both"      → csv_key used as prefix: {csv_key}_genomic, {csv_key}_processed
                             # "processed" → csv_key used directly
                             # "genomic"   → csv_key used directly
    description: str = ""


# Block definitions — source type and config key convention
BLOCK_DEFS: Dict[str, BlockMeta] = {
    "nonb": BlockMeta(
        name        = "nonb",
        source      = "both",
        csv_key     = "nonb",
        description = "Non-B DNA motif features (genomic locus + processed transcript)",
    ),
    "te": BlockMeta(
        name        = "te",
        source      = "both",
        csv_key     = "te",
        description = "Transposable element features (genomic locus + processed transcript)",
    ),
    "nonb2": BlockMeta(
        name        = "nonb2",
        source      = "processed",
        csv_key     = "nonb2",
        description = "RNA secondary structure (ScanFold2) + rG4 (rg4detector) features",
    ),
}


# ---------------------------------------------------------------------------
# NonB feature definitions (unchanged from original)
# ---------------------------------------------------------------------------

_NONB_MOTIF_TYPES: List[str] = ["apr", "dr", "gq", "ir", "mr", "str", "tri", "z"]

_NONB_MOTIF_STAT_TEMPLATE: List[Tuple[str, str, str]] = [
    ("hit_count",           "count",      "none"),
    ("total_length",        "abs_length", "robust"),
    ("max_length",          "abs_length", "robust"),
    ("mean_length",         "abs_length", "robust"),
    ("std_length",          "abs_length", "robust"),
    ("unique_length",       "abs_length", "robust"),
    ("gaps_mean",           "gap",        "robust"),
    ("gaps_median",         "gap",        "robust"),
    ("gaps_max",            "gap",        "robust"),
    ("gaps_min",            "gap",        "robust"),
    ("present",             "presence",   "none"),
    ("total_length_pct",    "rel_length", "minmax"),
    ("max_length_pct",      "rel_length", "minmax"),
    ("mean_length_pct",     "rel_length", "minmax"),
    ("std_length_pct",      "rel_length", "minmax"),
    ("unique_length_pct",   "rel_length", "minmax"),
    ("gaps_mean_pct",       "rel_length", "minmax"),
    ("gaps_median_pct",     "rel_length", "minmax"),
    ("gaps_max_pct",        "rel_length", "minmax"),
    ("gaps_min_pct",        "rel_length", "minmax"),
    ("hit_count_per_kb",    "density",    "robust"),
]

_NONB_GLOBAL_FEATURES: List[Tuple[str, str, str]] = [
    ("all_nonb_gaps_mean",      "gap",       "robust"),
    ("all_nonb_gaps_median",    "gap",       "robust"),
    ("all_nonb_gaps_max",       "gap",       "robust"),
    ("all_nonb_gaps_min",       "gap",       "robust"),
    ("any_nonb_present",        "presence",  "none"),
    ("motif_types_present",     "diversity", "none"),
    ("total_nonb_count",        "count",     "none"),
    ("total_nonb_coverage",     "abs_length","robust"),
    ("total_nonb_coverage_pct", "rel_length","minmax"),
    ("motif_diversity",         "diversity", "none"),
]


# ---------------------------------------------------------------------------
# TE feature definitions (unchanged from original)
# ---------------------------------------------------------------------------

_TE_QUALITY_FEATURES: List[Tuple[str, str, str]] = [
    ("te_min_sw_score",            "quality", "robust"),
    ("te_max_sw_score",            "quality", "robust"),
    ("te_mean_sw_score",           "quality", "robust"),
    ("te_min_divergence",          "quality", "robust"),
    ("te_max_divergence",          "quality", "robust"),
    ("te_mean_divergence",         "quality", "robust"),
    ("te_min_perc_del",            "quality", "robust"),
    ("te_max_perc_del",            "quality", "robust"),
    ("te_mean_perc_del",           "quality", "robust"),
    ("te_min_perc_ins",            "quality", "robust"),
    ("te_max_perc_ins",            "quality", "robust"),
    ("te_mean_perc_ins",           "quality", "robust"),
]

_TE_CORE_BASE_FEATURES: List[Tuple[str, str, str]] = [
    ("te_count",                   "count",      "none"),
    ("te_sum_hit_length",          "abs_length", "robust"),
    ("te_min_hit_length",          "abs_length", "robust"),
    ("te_mean_hit_length",         "abs_length", "robust"),
    ("te_max_hit_length",          "abs_length", "robust"),
    ("te_min_hit_reference_coverage",  "rel_length", "minmax"),
    ("te_mean_hit_reference_coverage", "rel_length", "minmax"),
    ("te_max_hit_reference_coverage",  "rel_length", "minmax"),
    ("te_sum_num_fragments",       "count",      "none"),
    ("te_mean_num_fragments",      "count",      "none"),
    ("te_max_num_fragments",       "count",      "none"),
    ("te_sum_fragmented",          "count",      "none"),
    ("te_unique_subfamilies",      "diversity",  "none"),
    ("te_unique_classes",          "diversity",  "none"),
    ("te_unique_families",         "diversity",  "none"),
    ("te_has_dna",                 "presence",   "none"),
    ("te_dna_count",               "count",      "none"),
    ("te_has_ervk",                "presence",   "none"),
    ("te_ervk_count",              "count",      "none"),
    ("te_has_ervl",                "presence",   "none"),
    ("te_ervl_count",              "count",      "none"),
    ("te_has_ervl-malr",           "presence",   "none"),
    ("te_ervl-malr_count",         "count",      "none"),
    ("te_has_erv1",                "presence",   "none"),
    ("te_erv1_count",              "count",      "none"),
    ("te_has_line",                "presence",   "none"),
    ("te_line_count",              "count",      "none"),
    ("te_has_ltr",                 "presence",   "none"),
    ("te_ltr_count",               "count",      "none"),
    ("te_has_ple",                 "presence",   "none"),
    ("te_ple_count",               "count",      "none"),
    ("te_has_rc",                  "presence",   "none"),
    ("te_rc_count",                "count",      "none"),
    ("te_has_retroposon",          "presence",   "none"),
    ("te_retroposon_count",        "count",      "none"),
    ("te_has_sine",                "presence",   "none"),
    ("te_sine_count",              "count",      "none"),
    ("te_has_srprna",              "presence",   "none"),
    ("te_srprna_count",            "count",      "none"),
    ("te_young_count",             "count",      "none"),
    ("te_ancient_count",           "count",      "none"),
    ("te_fragmented_ratio",        "rel_length", "minmax"),
    ("te_gaps_mean",               "gap",        "robust"),
    ("te_gaps_median",             "gap",        "robust"),
    ("te_gaps_max",                "gap",        "robust"),
    ("te_gaps_min",                "gap",        "robust"),
]

_TE_LCTR_BASE_FEATURES: List[Tuple[str, str, str]] = [
    ("lctr_count",                     "count",      "none"),
    ("lctr_total_length",              "abs_length", "robust"),
    ("lctr_mean_length",               "abs_length", "robust"),
    ("lctr_sum_num_fragments",         "count",      "none"),
    ("lctr_mean_num_fragments",        "count",      "none"),
    ("lctr_has_low_complexity",        "presence",   "none"),
    ("lctr_low_complexity_count",      "count",      "none"),
    ("lctr_has_simple_repeat",         "presence",   "none"),
    ("lctr_simple_repeat_count",       "count",      "none"),
    ("lctr_has_satellite",             "presence",   "none"),
    ("lctr_satellite_count",           "count",      "none"),
    ("lctr_gaps_mean",                 "gap",        "robust"),
    ("lctr_gaps_median",               "gap",        "robust"),
    ("lctr_gaps_max",                  "gap",        "robust"),
    ("lctr_gaps_min",                  "gap",        "robust"),
]

_TE_PSEUDO_FEATURES: List[Tuple[str, str, str]] = [
    ("pseudo_min_sw_score",            "quality",    "robust"),
    ("pseudo_max_sw_score",            "quality",    "robust"),
    ("pseudo_mean_sw_score",           "quality",    "robust"),
    ("pseudo_min_divergence",          "quality",    "robust"),
    ("pseudo_max_divergence",          "quality",    "robust"),
    ("pseudo_mean_divergence",         "quality",    "robust"),
    ("pseudo_min_perc_del",            "quality",    "robust"),
    ("pseudo_max_perc_del",            "quality",    "robust"),
    ("pseudo_mean_perc_del",           "quality",    "robust"),
    ("pseudo_min_perc_ins",            "quality",    "robust"),
    ("pseudo_max_perc_ins",            "quality",    "robust"),
    ("pseudo_mean_perc_ins",           "quality",    "robust"),
    ("pseudo_count",                   "count",      "none"),
    ("pseudo_sum_hit_length",          "abs_length", "robust"),
    ("pseudo_min_hit_length",          "abs_length", "robust"),
    ("pseudo_mean_hit_length",         "abs_length", "robust"),
    ("pseudo_max_hit_length",          "abs_length", "robust"),
    ("pseudo_min_hit_reference_coverage",  "rel_length", "minmax"),
    ("pseudo_mean_hit_reference_coverage", "rel_length", "minmax"),
    ("pseudo_max_hit_reference_coverage",  "rel_length", "minmax"),
    ("pseudo_sum_num_fragments",       "count",      "none"),
    ("pseudo_mean_num_fragments",      "count",      "none"),
    ("pseudo_max_num_fragments",       "count",      "none"),
    ("pseudo_sum_fragmented",          "count",      "none"),
    ("pseudo_unique_subfamilies",      "diversity",  "none"),
    ("pseudo_unique_classes",          "diversity",  "none"),
    ("pseudo_unique_families",         "diversity",  "none"),
    ("pseudo_has_rrna",                "presence",   "none"),
    ("pseudo_rrna_count",              "count",      "none"),
    ("pseudo_has_scrna",               "presence",   "none"),
    ("pseudo_scrna_count",             "count",      "none"),
    ("pseudo_has_snrna",               "presence",   "none"),
    ("pseudo_snrna_count",             "count",      "none"),
    ("pseudo_has_trna",                "presence",   "none"),
    ("pseudo_trna_count",              "count",      "none"),
    ("pseudo_ancient_count",           "count",      "none"),
    ("pseudogene_gaps_mean",           "gap",        "robust"),
    ("pseudogene_gaps_median",         "gap",        "robust"),
    ("pseudogene_gaps_max",            "gap",        "robust"),
    ("pseudogene_gaps_min",            "gap",        "robust"),
]

_TE_UNKNOWN_BASE_FEATURES: List[Tuple[str, str, str]] = [
    ("unknown_count",              "count",      "none"),
    ("unknown_total_length",       "abs_length", "robust"),
    ("unknown_mean_length",        "abs_length", "robust"),
    ("unknown_sum_num_fragments",  "count",      "none"),
    ("unknown_mean_num_fragments", "count",      "none"),
    ("unknown_gaps_mean",          "gap",        "robust"),
    ("unknown_gaps_median",        "gap",        "robust"),
    ("unknown_gaps_max",           "gap",        "robust"),
    ("unknown_gaps_min",           "gap",        "robust"),
]

_TE_GLOBAL_BASE_FEATURES: List[Tuple[str, str, str]] = [
    ("global_rm_count",            "count",      "none"),
    ("global_rm_total_length",     "abs_length", "robust"),
    ("global_gaps_mean",           "gap",        "robust"),
    ("global_gaps_median",         "gap",        "robust"),
    ("global_gaps_max",            "gap",        "robust"),
    ("global_gaps_min",            "gap",        "robust"),
]

_TE_CORE_PCT_FEATURES: List[Tuple[str, str, str]] = [
    ("te_sum_hit_length_pct",      "rel_length", "minmax"),
    ("te_min_hit_length_pct",      "rel_length", "minmax"),
    ("te_mean_hit_length_pct",     "rel_length", "minmax"),
    ("te_max_hit_length_pct",      "rel_length", "minmax"),
    ("te_gaps_mean_pct",           "rel_length", "minmax"),
    ("te_gaps_median_pct",         "rel_length", "minmax"),
    ("te_gaps_max_pct",            "rel_length", "minmax"),
    ("te_gaps_min_pct",            "rel_length", "minmax"),
]

_TE_LCTR_PCT_FEATURES: List[Tuple[str, str, str]] = [
    ("lctr_total_length_pct",      "rel_length", "minmax"),
    ("lctr_mean_length_pct",       "rel_length", "minmax"),
]

_TE_UNKNOWN_PCT_FEATURES: List[Tuple[str, str, str]] = [
    ("unknown_total_length_pct",   "rel_length", "minmax"),
    ("unknown_mean_length_pct",    "rel_length", "minmax"),
]

_TE_GLOBAL_PCT_FEATURES: List[Tuple[str, str, str]] = [
    ("global_rm_total_length_pct", "rel_length", "minmax"),
    ("global_gaps_mean_pct",       "rel_length", "minmax"),
    ("global_gaps_median_pct",     "rel_length", "minmax"),
]

_TE_DENSITY_FEATURES: List[Tuple[str, str, str, str]] = [
    ("te_count_per_kb",                "density", "robust", "TE_CORE"),
    ("lctr_count_per_kb",              "density", "robust", "TE_LCTR"),
    ("unknown_count_per_kb",           "density", "robust", "TE_UNKNOWN"),
    ("global_rm_count_per_kb",         "density", "robust", "TE_GLOBAL"),
    ("te_dna_count_per_kb",            "density", "robust", "TE_CORE"),
    ("te_ervk_count_per_kb",           "density", "robust", "TE_CORE"),
    ("te_ervl_count_per_kb",           "density", "robust", "TE_CORE"),
    ("te_ervl-malr_count_per_kb",      "density", "robust", "TE_CORE"),
    ("te_erv1_count_per_kb",           "density", "robust", "TE_CORE"),
    ("te_line_count_per_kb",           "density", "robust", "TE_CORE"),
    ("te_ltr_count_per_kb",            "density", "robust", "TE_CORE"),
    ("te_ple_count_per_kb",            "density", "robust", "TE_CORE"),
    ("te_rc_count_per_kb",             "density", "robust", "TE_CORE"),
    ("te_retroposon_count_per_kb",     "density", "robust", "TE_CORE"),
    ("te_sine_count_per_kb",           "density", "robust", "TE_CORE"),
    ("te_srprna_count_per_kb",         "density", "robust", "TE_CORE"),
    ("te_young_count_per_kb",          "density", "robust", "TE_CORE"),
    ("te_ancient_count_per_kb",        "density", "robust", "TE_CORE"),
    ("lctr_low_complexity_count_per_kb","density","robust", "TE_LCTR"),
    ("lctr_simple_repeat_count_per_kb","density", "robust", "TE_LCTR"),
    ("lctr_satellite_count_per_kb",    "density", "robust", "TE_LCTR"),
    ("pseudo_count_per_kb",            "density", "robust", "TE_PSEUDO"),
    ("pseudo_rrna_count_per_kb",       "density", "robust", "TE_PSEUDO"),
    ("pseudo_scrna_count_per_kb",      "density", "robust", "TE_PSEUDO"),
    ("pseudo_snrna_count_per_kb",      "density", "robust", "TE_PSEUDO"),
    ("pseudo_trna_count_per_kb",       "density", "robust", "TE_PSEUDO"),
    ("pseudo_ancient_count_per_kb",    "density", "robust", "TE_PSEUDO"),
]


# ---------------------------------------------------------------------------
# NonB2 feature definitions (RNA secondary structure + rG4)
# ---------------------------------------------------------------------------
#
# Source: "processed" (RNA/transcript-level features only, no genomic version)
#
# Columns excluded from features:
#   - length          : transcript length metadata, length confound, excluded
#   - n_paired        : absolute count, length-dependent, use frac_paired instead
#   - n_competition   : absolute count, use frac_competition instead
#   - n_z_lt_minus1   : absolute count, use frac_z_lt_minus1 instead
#   - n_z_lt_minus2   : absolute count, use frac_z_lt_minus2 instead
#   - n_*_bin{1..10}  : absolute bin counts, length-dependent,
#                       use prop_*_bin{1..10} (proportions) instead
#
# rg4_peak_rel_position sentinel: -1.0 when has_peak=False (no peaks detected).
# Imputed to 0.5 (neutral midpoint) in prepare_features.py before scaling.
# has_peak (bool) separately encodes peak absence.

# ── RG4 subgroup (rg4detector) ─────────────────────────────────────────────
_RG4_FEATURES: List[Tuple[str, str, str]] = [
    ("rg4_peak_count",        "count",    "none"),
    ("rg4_peak_density",      "density",  "robust"),
    ("rg4_peak_rel_position", "fraction", "minmax"),   # imputed sentinel → 0.5
    ("has_peak",              "presence", "none"),      # bool → 0/1 cast before scaling
]

# ── SS_STAB subgroup (ScanFold2 stability) ────────────────────────────────
# Per-nucleotide average of windowed Z-score, MFE, ED — length-invariant
_SS_STAB_FEATURES: List[Tuple[str, str, str]] = [
    ("avgZ_min",    "stability", "robust"),
    ("avgZ_mean",   "stability", "robust"),
    ("avgZ_median", "stability", "robust"),
    ("avgZ_max",    "stability", "robust"),
    ("avgMFE_min",  "stability", "robust"),
    ("avgMFE_mean", "stability", "robust"),
    ("avgMFE_median","stability","robust"),
    ("avgMFE_max",  "stability", "robust"),
    ("avgED_min",   "stability", "robust"),
    ("avgED_mean",  "stability", "robust"),
    ("avgED_median","stability", "robust"),
    ("avgED_max",   "stability", "robust"),
]

# ── SS_COUNT subgroup (ScanFold2 fractions — length-normalised) ───────────
# n_ counterparts excluded (length-dependent)
_SS_COUNT_FEATURES: List[Tuple[str, str, str]] = [
    ("frac_paired",      "fraction", "minmax"),
    ("frac_competition", "fraction", "minmax"),
    ("frac_z_lt_minus1", "fraction", "minmax"),
    ("frac_z_lt_minus2", "fraction", "minmax"),
]

# ── SS_RELPOS subgroup (ScanFold2 relative positions) ─────────────────────
_SS_RELPOS_FEATURES: List[Tuple[str, str, str]] = [
    ("rel_pos_avgZ_min",   "fraction", "minmax"),
    ("rel_pos_avgMFE_min", "fraction", "minmax"),
    ("rel_pos_avgED_min",  "fraction", "minmax"),
]

# ── SS_BINNED subgroup (ScanFold2 per-bin proportions) ────────────────────
# prop_ versions kept (length-normalised); n_ versions excluded
_N_BINS = 10
_SS_BINNED_FEATURES: List[Tuple[str, str, str]] = [
    (f"prop_{feat}_bin{b}", "binned_prop", "minmax")
    for feat in ["paired", "competition", "z_lt_minus1", "z_lt_minus2"]
    for b in range(1, _N_BINS + 1)
]


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def _build_nonb() -> List[FeatureMeta]:
    nonb: List[FeatureMeta] = []
    for motif in _NONB_MOTIF_TYPES:
        subgroup = motif.upper()
        for suffix, category, scale in _NONB_MOTIF_STAT_TEMPLATE:
            nonb.append(FeatureMeta(
                name=f"{motif}_{suffix}", block="nonb",
                subgroup=subgroup, category=category, scale=scale,
            ))
    for name, category, scale in _NONB_GLOBAL_FEATURES:
        nonb.append(FeatureMeta(
            name=name, block="nonb",
            subgroup="GLOBAL", category=category, scale=scale,
        ))
    return nonb


def _build_te() -> List[FeatureMeta]:
    te: List[FeatureMeta] = []
    simple_blocks = [
        ("TE_QUALITY",  _TE_QUALITY_FEATURES),
        ("TE_CORE",     _TE_CORE_BASE_FEATURES),
        ("TE_LCTR",     _TE_LCTR_BASE_FEATURES),
        ("TE_PSEUDO",   _TE_PSEUDO_FEATURES),
        ("TE_UNKNOWN",  _TE_UNKNOWN_BASE_FEATURES),
        ("TE_GLOBAL",   _TE_GLOBAL_BASE_FEATURES),
    ]
    for subgroup, feat_list in simple_blocks:
        for name, category, scale in feat_list:
            te.append(FeatureMeta(
                name=name, block="te",
                subgroup=subgroup, category=category, scale=scale,
            ))
    pct_blocks = [
        ("TE_CORE",    _TE_CORE_PCT_FEATURES),
        ("TE_LCTR",    _TE_LCTR_PCT_FEATURES),
        ("TE_UNKNOWN", _TE_UNKNOWN_PCT_FEATURES),
        ("TE_GLOBAL",  _TE_GLOBAL_PCT_FEATURES),
    ]
    for subgroup, feat_list in pct_blocks:
        for name, category, scale in feat_list:
            te.append(FeatureMeta(
                name=name, block="te",
                subgroup=subgroup, category=category, scale=scale,
            ))
    for name, category, scale, subgroup in _TE_DENSITY_FEATURES:
        te.append(FeatureMeta(
            name=name, block="te",
            subgroup=subgroup, category=category, scale=scale,
        ))
    return te


def _build_nonb2() -> List[FeatureMeta]:
    nonb2: List[FeatureMeta] = []
    subgroup_blocks = [
        ("RG4",       _RG4_FEATURES),
        ("SS_STAB",   _SS_STAB_FEATURES),
        ("SS_COUNT",  _SS_COUNT_FEATURES),
        ("SS_RELPOS", _SS_RELPOS_FEATURES),
        ("SS_BINNED", _SS_BINNED_FEATURES),
    ]
    for subgroup, feat_list in subgroup_blocks:
        for name, category, scale in feat_list:
            nonb2.append(FeatureMeta(
                name=name, block="nonb2",
                subgroup=subgroup, category=category, scale=scale,
            ))
    return nonb2


# ---------------------------------------------------------------------------
# FeatureRegistry — the public API
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """
    Indexed registry of all feature blocks (NonB, TE, NonB2).

    The registry is the single source of truth for:
      - Feature names and their order in each CSV
      - Subgroup membership
      - Block source type ("both" | "processed" | "genomic")
      - Scaling strategy per feature category

    New blocks can be added by:
      1. Defining feature lists above
      2. Adding a builder call in __init__
      3. Adding a BlockMeta entry in BLOCK_DEFS

    Usage
    -----
    >>> reg = FeatureRegistry()

    # All blocks in order
    >>> reg.block_names          # ["nonb", "te", "nonb2"]
    >>> reg.block_source("te")   # "both"

    # Subgroup → block lookup
    >>> reg.token_block("RG4")   # "nonb2"
    >>> reg.token_block("APR")   # "nonb"

    # All subgroup names in token order
    >>> reg.all_subgroups        # ["APR", "DR", ..., "TE_CORE", ..., "RG4", ...]

    # Existing block-specific APIs still work
    >>> reg.nonb_indices_for_subgroup("GQ")
    >>> reg.te_indices_for_subgroup("TE_CORE")
    >>> reg.nonb2_indices_for_subgroup("SS_STAB")

    # Validate against actual CSV columns
    >>> reg.validate(nonb_columns=..., te_columns=..., nonb2_columns=...)
    """

    def __init__(self) -> None:
        self._nonb  = _build_nonb()
        self._te    = _build_te()
        self._nonb2 = _build_nonb2()

        # Ordered block registry: insertion order = token concatenation order
        # (nonb tokens | te tokens | nonb2 tokens)
        self._blocks: Dict[str, List[FeatureMeta]] = {
            "nonb":  self._nonb,
            "te":    self._te,
            "nonb2": self._nonb2,
        }

        # Name → index maps per block
        self._nonb_name_to_idx:  Dict[str, int] = {m.name: i for i, m in enumerate(self._nonb)}
        self._te_name_to_idx:    Dict[str, int] = {m.name: i for i, m in enumerate(self._te)}
        self._nonb2_name_to_idx: Dict[str, int] = {m.name: i for i, m in enumerate(self._nonb2)}

        # Subgroup → block lookup (built once)
        self._sg_to_block: Dict[str, str] = {}
        for block_name, features in self._blocks.items():
            for m in features:
                if m.subgroup in self._sg_to_block:
                    assert self._sg_to_block[m.subgroup] == block_name, (
                        f"Subgroup '{m.subgroup}' appears in multiple blocks: "
                        f"{self._sg_to_block[m.subgroup]} and {block_name}"
                    )
                self._sg_to_block[m.subgroup] = block_name

    # ------------------------------------------------------------------
    # Block-level API (new)
    # ------------------------------------------------------------------

    @property
    def block_names(self) -> List[str]:
        """Ordered list of block names — matches token concatenation order."""
        return list(self._blocks.keys())

    def block_source(self, block_name: str) -> str:
        """Return source type for a block: 'both' | 'processed' | 'genomic'."""
        return BLOCK_DEFS[block_name].source

    def block_features(self, block_name: str) -> List[FeatureMeta]:
        """Return ordered FeatureMeta list for a block."""
        return self._blocks[block_name]

    def block_dim(self, block_name: str) -> int:
        """Return feature dimension for a block."""
        return len(self._blocks[block_name])

    def block_subgroups(self, block_name: str) -> List[str]:
        """Return ordered list of subgroup names for a block."""
        seen: Dict[str, None] = {}
        for m in self._blocks[block_name]:
            seen[m.subgroup] = None
        return list(seen)

    def token_block(self, subgroup: str) -> str:
        """Return the block name that contains a given subgroup."""
        if subgroup not in self._sg_to_block:
            raise KeyError(f"Subgroup '{subgroup}' not found in any block. "
                           f"Known subgroups: {list(self._sg_to_block)}")
        return self._sg_to_block[subgroup]

    @property
    def all_subgroups(self) -> List[str]:
        """All subgroup names in token order (nonb | te | nonb2)."""
        result: Dict[str, None] = {}
        for block_name in self.block_names:
            for sg in self.block_subgroups(block_name):
                result[sg] = None
        return list(result)

    @property
    def total_tokens(self) -> int:
        """Total number of subgroup tokens across all blocks."""
        return len(self.all_subgroups)

    def indices_for_subgroup(self, subgroup: str) -> List[int]:
        """
        Return feature indices for a subgroup within its block's feature vector.
        Equivalent to the block-specific methods below.
        """
        block = self.token_block(subgroup)
        features = self._blocks[block]
        return [i for i, m in enumerate(features) if m.subgroup == subgroup]

    def scale_strategy(self, block_name: str) -> Dict[str, str]:
        """Return {category: scale_strategy} for a block."""
        result: Dict[str, str] = {}
        for m in self._blocks[block_name]:
            if m.category in result and result[m.category] != m.scale:
                raise ValueError(
                    f"Block '{block_name}' category '{m.category}' has conflicting "
                    f"scale strategies: {result[m.category]} vs {m.scale} "
                    f"(feature: {m.name})"
                )
            result[m.category] = m.scale
        return result

    # ------------------------------------------------------------------
    # Backward-compatible properties (unchanged)
    # ------------------------------------------------------------------

    @property
    def nonb_features(self) -> List[FeatureMeta]:
        return self._nonb

    @property
    def te_features(self) -> List[FeatureMeta]:
        return self._te

    @property
    def nonb2_features(self) -> List[FeatureMeta]:
        return self._nonb2

    @property
    def nonb_dim(self) -> int:
        return len(self._nonb)

    @property
    def te_dim(self) -> int:
        return len(self._te)

    @property
    def nonb2_dim(self) -> int:
        return len(self._nonb2)

    @property
    def nonb_subgroups(self) -> List[str]:
        return self.block_subgroups("nonb")

    @property
    def te_subgroups(self) -> List[str]:
        return self.block_subgroups("te")

    @property
    def nonb2_subgroups(self) -> List[str]:
        return self.block_subgroups("nonb2")

    # ------------------------------------------------------------------
    # Index lookups — NonB (unchanged)
    # ------------------------------------------------------------------

    def nonb_indices_for_subgroup(self, subgroup: str) -> List[int]:
        return [i for i, m in enumerate(self._nonb) if m.subgroup == subgroup]

    def nonb_indices_for_category(self, category: str) -> List[int]:
        return [i for i, m in enumerate(self._nonb) if m.category == category]

    def nonb_category_indices(self) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        for i, m in enumerate(self._nonb):
            result.setdefault(m.category, []).append(i)
        return result

    def nonb_subgroup_category_indices(self) -> Dict[str, Dict[str, List[int]]]:
        result: Dict[str, Dict[str, List[int]]] = {}
        for i, m in enumerate(self._nonb):
            result.setdefault(m.subgroup, {}).setdefault(m.category, []).append(i)
        return result

    def nonb_scale_strategy(self) -> Dict[str, str]:
        return self.scale_strategy("nonb")

    # ------------------------------------------------------------------
    # Index lookups — TE (unchanged)
    # ------------------------------------------------------------------

    def te_indices_for_subgroup(self, subgroup: str) -> List[int]:
        return [i for i, m in enumerate(self._te) if m.subgroup == subgroup]

    def te_indices_for_category(self, category: str) -> List[int]:
        return [i for i, m in enumerate(self._te) if m.category == category]

    def te_category_indices(self) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        for i, m in enumerate(self._te):
            result.setdefault(m.category, []).append(i)
        return result

    def te_subgroup_category_indices(self) -> Dict[str, Dict[str, List[int]]]:
        result: Dict[str, Dict[str, List[int]]] = {}
        for i, m in enumerate(self._te):
            result.setdefault(m.subgroup, {}).setdefault(m.category, []).append(i)
        return result

    def te_scale_strategy(self) -> Dict[str, str]:
        return self.scale_strategy("te")

    # ------------------------------------------------------------------
    # Index lookups — NonB2 (new)
    # ------------------------------------------------------------------

    def nonb2_indices_for_subgroup(self, subgroup: str) -> List[int]:
        return [i for i, m in enumerate(self._nonb2) if m.subgroup == subgroup]

    def nonb2_indices_for_category(self, category: str) -> List[int]:
        return [i for i, m in enumerate(self._nonb2) if m.category == category]

    def nonb2_category_indices(self) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        for i, m in enumerate(self._nonb2):
            result.setdefault(m.category, []).append(i)
        return result

    def nonb2_scale_strategy(self) -> Dict[str, str]:
        return self.scale_strategy("nonb2")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        nonb_columns:   Optional[List[str]] = None,
        te_columns:     Optional[List[str]] = None,
        nonb2_columns:  Optional[List[str]] = None,
        strict:         bool = True,
    ) -> None:
        """
        Validate the registry against actual CSV column lists.
        Any block can be omitted (None) to skip its validation.

        Raises ValueError if strict=True and mismatches are found.
        """
        checks = [
            ("nonb",  self._nonb,  nonb_columns),
            ("te",    self._te,    te_columns),
            ("nonb2", self._nonb2, nonb2_columns),
        ]
        errors: List[str] = []

        for block_name, features, columns in checks:
            if columns is None:
                continue
            registry_names = [m.name for m in features]
            if registry_names == columns:
                continue
            missing = set(registry_names) - set(columns)
            extra   = set(columns) - set(registry_names)
            order_only = not missing and not extra

            if missing:
                errors.append(
                    f"{block_name}: in registry but not in CSV: {sorted(missing)}")
            if extra:
                errors.append(
                    f"{block_name}: in CSV but not in registry: {sorted(extra)}")
            if order_only:
                errors.append(
                    f"{block_name}: feature ORDER differs — indices will be wrong")

        if errors:
            msg = "\n".join(errors)
            if strict:
                raise ValueError(f"FeatureRegistry validation failed:\n{msg}")
            else:
                import warnings
                warnings.warn(f"FeatureRegistry validation warnings:\n{msg}")
        else:
            validated = [b for b, _, c in checks if c is not None]
            dims = {b: self.block_dim(b) for b in validated}
            print(f"FeatureRegistry validation passed: {dims}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["FeatureRegistry summary", "=" * 65]
        for block_name in self.block_names:
            meta = BLOCK_DEFS[block_name]
            dim  = self.block_dim(block_name)
            sgs  = self.block_subgroups(block_name)
            lines.append(f"\n{block_name} block — {dim} features  "
                         f"[source={meta.source}]")
            lines.append(f"  {meta.description}")
            lines.append(f"  Subgroups: {sgs}")
            for sg in sgs:
                idx = self.indices_for_subgroup(sg)
                lines.append(f"    {sg:15s}: {len(idx):3d} features  "
                             f"(indices {idx[0]}–{idx[-1]})")
            cat_counts: Dict[str, int] = {}
            for m in self.block_features(block_name):
                cat_counts[m.category] = cat_counts.get(m.category, 0) + 1
            lines.append(f"  Categories: {cat_counts}")

        lines.append(f"\nTotal tokens : {self.total_tokens}  "
                     f"({' + '.join(str(len(self.block_subgroups(b))) for b in self.block_names)} "
                     f"= {' + '.join(self.block_names)})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

REGISTRY = FeatureRegistry()


# ---------------------------------------------------------------------------
# Display name maps
#
# Internal code, config keys, CSV column names, and checkpoint parameter
# names use "te" / "TE_*" as prefixes. These maps translate to the
# scientifically accurate display labels used in plots and reports:
# "TE" (Transposable Elements) is replaced by "REP" (Repetitive Elements)
# since several subgroups (REP_LCTR, REP_PSEUDO) are not transposable
# elements — LCTR covers low-complexity repeats and tandem repeats,
# PSEUDO covers pseudogene-derived sequences.
# ---------------------------------------------------------------------------

# Block internal key → display label
BLOCK_DISPLAY_NAMES: dict[str, str] = {
    "nonb":  "NonB",
    "te":    "REP",
    "nonb2": "NonB2",
}

# Subgroup internal name → display label
SUBGROUP_DISPLAY_NAMES: dict[str, str] = {
    # REP block (formerly TE)
    "TE_CORE":    "REP_CORE",
    "TE_LCTR":    "REP_LCTR",
    "TE_QUALITY": "REP_QUALITY",
    "TE_PSEUDO":  "REP_PSEUDO",
    "TE_UNKNOWN": "REP_UNKNOWN",
    "TE_GLOBAL":  "REP_GLOBAL",
    # NonB, NonB2, and any future blocks default to their own name (see helper)
}


def display_block(block_name: str) -> str:
    """Return the display label for a block internal key."""
    return BLOCK_DISPLAY_NAMES.get(block_name, block_name.upper())


def display_subgroup(subgroup: str) -> str:
    """Return the display label for a subgroup internal name."""
    return SUBGROUP_DISPLAY_NAMES.get(subgroup, subgroup)


def display_subgroups(subgroups) -> list[str]:
    """Map a list of internal subgroup names to their display labels."""
    return [display_subgroup(sg) for sg in subgroups]



# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reg = FeatureRegistry()
    print(reg.summary())

    print("\nBlock names  :", reg.block_names)
    print("All subgroups:", reg.all_subgroups)
    print("Total tokens :", reg.total_tokens)

    print("\ntoken_block('APR')    :", reg.token_block("APR"))
    print("token_block('TE_CORE'):", reg.token_block("TE_CORE"))
    print("token_block('RG4')    :", reg.token_block("RG4"))
    print("token_block('SS_STAB'):", reg.token_block("SS_STAB"))

    print("\nNonB2 subgroups:", reg.nonb2_subgroups)
    print("NonB2 dim      :", reg.nonb2_dim)
    print("NonB2 RG4 indices:", reg.nonb2_indices_for_subgroup("RG4"))
    print("NonB2 SS_BINNED indices (first 5):",
          reg.nonb2_indices_for_subgroup("SS_BINNED")[:5])

    # Backward compatibility
    assert reg.nonb_dim  == 178, f"Expected 178, got {reg.nonb_dim}"
    assert reg.te_dim    == 170, f"Expected 170, got {reg.te_dim}"
    assert reg.nonb2_dim ==  63, f"Expected 63,  got {reg.nonb2_dim}"
    assert reg.total_tokens == 20, f"Expected 20 tokens, got {reg.total_tokens}"
    print(f"\nDimension assertions passed: "
          f"NonB={reg.nonb_dim}, TE={reg.te_dim}, "
          f"NonB2={reg.nonb2_dim}, tokens={reg.total_tokens}")

    print("\nNonB2 scale strategies:", reg.nonb2_scale_strategy())
    print("\nAll self-tests passed.")