"""NASA TESS Exoplanet Vetting Domain (`ExoplanetTriageDomain`).

Flagship scientific domain example for ARES ("Planet Hunter vs. Cosmic Impostors"):
- Operates over real NASA TESS SPOC 2-minute cadence photometry (`lightcurves`),
  Gaia/TIC host stellar parameters (`stellar_catalog`), and real 11x11 TESS CCD
  Target Pixel File difference stamps (`pixel_stamps`) via `DuckDBBackend`.
- Enforces 4 independent physical & spatial checks plus a compound confirmation gate:
    1. `periodic_transit_snr` (`verify_periodic_transit_snr`)
    2. `odd_even_eclipse_clean` (`verify_odd_even_eclipse_symmetry`)
    3. `keplerian_density_match` (`verify_keplerian_density_consistency`)
    4. `ccd_centroid_lock` (`verify_ccd_centroid_lock`)
    5. `confirmed_exoplanet_target` (`verify_exoplanet_confirmation`)
- Emits deterministic SHA-256 2x2 dark-mode NASA Observatory PNG diagnostic cards
  (`examples/exoplanet_triage/artifacts.py`) for multimodal `gemini-4-carbon` context synthesis.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ares.adapters.duckdb_backend import DuckDBBackend
from ares.contracts import (
    ArtifactRef,
    ExecutionBackend,
    GroundTruth,
    InvestigationBoard,
    NeedleSpec,
    Note,
    SearchQuestion,
    VerifierRegistry,
    VerifierResult,
)
from examples.exoplanet_triage.artifacts import render_exoplanet_diagnostic_card

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STAGED_DIR = REPO_ROOT / "data" / "tess_staged"

EXOPLANET_PIVOTAL_FIELDS: list[str] = [
    "periodic_transit_snr",
    "odd_even_eclipse_clean",
    "keplerian_density_match",
    "ccd_centroid_lock",
    "confirmed_exoplanet_target",
]

EXOPLANET_REQUIRED_SIGNALS: tuple[str, ...] = (
    "periodic_transit_snr",
    "odd_even_eclipse_clean",
    "keplerian_density_match",
    "ccd_centroid_lock",
)

_VERIFIER_FIELD_MAP: dict[str, str] = {
    "verify_periodic_transit_snr": "periodic_transit_snr",
    "verify_odd_even_eclipse_symmetry": "odd_even_eclipse_clean",
    "verify_keplerian_density_consistency": "keplerian_density_match",
    "verify_ccd_centroid_lock": "ccd_centroid_lock",
    "verify_exoplanet_confirmation": "confirmed_exoplanet_target",
}

FALLBACK_CANDIDATE_TICS: tuple[str, ...] = (
    "TIC-261136679",
    "TIC-307210830",
    "TIC-25155310",
    "TIC-149603524",
    "TIC-231663901",
    "TIC-278866211",
    "TIC-16740101",
    "TIC-349827430",
    "TIC-260304296",
)

LIGHTCURVES_SCHEMA = pa.schema(
    [
        ("tic_id", pa.string()),
        ("sector", pa.int32()),
        ("cadence_no", pa.int32()),
        ("btjd", pa.float64()),
        ("pdcsap_flux", pa.float64()),
        ("rel_flux_ppm", pa.float64()),
        ("flux_err_ppm", pa.float64()),
        ("mom_centr1", pa.float64()),
        ("mom_centr2", pa.float64()),
        ("quality_flag", pa.int32()),
    ]
)

STELLAR_CATALOG_SCHEMA = pa.schema(
    [
        ("tic_id", pa.string()),
        ("sector", pa.int32()),
        ("tmag", pa.float64()),
        ("teff_k", pa.float64()),
        ("stellar_radius_rsun", pa.float64()),
        ("stellar_mass_msun", pa.float64()),
        ("stellar_density_gcc", pa.float64()),
        ("crowding_metric", pa.float64()),
        ("flux_fraction", pa.float64()),
        ("ra_deg", pa.float64()),
        ("dec_deg", pa.float64()),
        ("candidate_period_days", pa.float64()),
        ("candidate_t0_btjd", pa.float64()),
        ("candidate_duration_hours", pa.float64()),
    ]
)

PIXEL_STAMPS_SCHEMA = pa.schema(
    [
        ("tic_id", pa.string()),
        ("sector", pa.int32()),
        ("row_idx", pa.int32()),
        ("col_idx", pa.int32()),
        ("mean_out_flux", pa.float64()),
        ("mean_in_flux", pa.float64()),
        ("diff_flux", pa.float64()),
        ("diff_snr", pa.float64()),
        ("aperture_mask", pa.int32()),
        ("target_row", pa.float64()),
        ("target_col", pa.float64()),
    ]
)


def _parquet_glob(data_dir: str, table: str) -> str:
    return str(Path(data_dir) / table / "*.parquet").replace("'", "''")


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _extract_candidate_tics(note: Note) -> str | list[str]:
    candidates = note.evidence.get("candidates")
    if note.row_count == 0 or not isinstance(candidates, list) or len(candidates) == 0:
        return "none"
    tics = [
        str(c["tic_id"])
        for c in candidates
        if isinstance(c, dict) and c.get("tic_id") is not None
    ]
    if not tics:
        return "none"
    if len(tics) == 1:
        return tics[0]
    return tics


def extract_exoplanet_note_fields(note: Note) -> dict[str, Any] | None:
    """Deterministic verifier-note -> canonical board field extractor for TESS vetting."""
    vname = note.verifier_name
    if vname not in _VERIFIER_FIELD_MAP:
        return None
    fname = _VERIFIER_FIELD_MAP[vname]
    if vname == "verify_exoplanet_confirmation":
        confirmed = note.evidence.get("confirmed_planets")
        if isinstance(confirmed, list):
            if len(confirmed) == 0 or note.row_count == 0:
                return {fname: "none"}
            if len(confirmed) == 1:
                return {fname: str(confirmed[0])}
            return {fname: [str(x) for x in confirmed]}
    return {fname: _extract_candidate_tics(note)}


def _build_fallback_menu() -> list[NeedleSpec]:
    """Deterministic 9-case TESS schema menu used when `data/tess_staged/` is absent."""
    return [
        NeedleSpec(
            case_id="tess_mc_01_rocky_planet",
            category="multi_clue_join",
            scenario_type="confirmed_exoplanet",
            target_entity="TIC-261136679",
            subtlety="medium",
            description="Confirmed short-period exoplanet passing SNR, odd/even, Keplerian density, and on-target CCD lock.",
            expected_decisions={
                "confirmed_planet_tic": "TIC-261136679",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            seed=501,
            metadata={"impostor_mode": "none", "toi_label": "CP"},
        ),
        NeedleSpec(
            case_id="tess_mc_02_warm_subneptune",
            category="multi_clue_join",
            scenario_type="confirmed_exoplanet",
            target_entity="TIC-307210830",
            subtlety="high",
            description="Subtle warm sub-Neptune passing all four photometric, orbital, and pixel-centroid checks.",
            expected_decisions={
                "confirmed_planet_tic": "TIC-307210830",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            seed=502,
            metadata={"impostor_mode": "none", "toi_label": "CP"},
        ),
        NeedleSpec(
            case_id="tess_mc_03_hot_jupiter",
            category="multi_clue_join",
            scenario_type="confirmed_exoplanet",
            target_entity="TIC-25155310",
            subtlety="low",
            description="High-SNR confirmed giant planet with clean symmetric transits and bullseye CCD centroid.",
            expected_decisions={
                "confirmed_planet_tic": "TIC-25155310",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            seed=503,
            metadata={"impostor_mode": "none", "toi_label": "KP"},
        ),
        NeedleSpec(
            case_id="tess_decoy_01_bg_binary_shift",
            category="look_alike_decoy",
            scenario_type="centroid_shift_impostor",
            target_entity="TIC-261136679",
            decoy_entity="TIC-231663901",
            subtlety="high",
            description="Loud background eclipsing binary decoy (5,800 ppm dip) shifts 2.1 px off-target on CCD; subtle real planet locks on target.",
            expected_decisions={
                "confirmed_planet_tic": "TIC-261136679",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            seed=601,
            metadata={"impostor_mode": "centroid_shift", "toi_label": "FP_vs_CP"},
        ),
        NeedleSpec(
            case_id="tess_decoy_02_odd_even_binary",
            category="look_alike_decoy",
            scenario_type="odd_even_impostor",
            target_entity="TIC-307210830",
            decoy_entity="TIC-278866211",
            subtlety="high",
            description="On-target eclipsing binary decoy passes CCD lock & density but fails odd/even eclipse depth symmetry; real planet passes all 4.",
            expected_decisions={
                "confirmed_planet_tic": "TIC-307210830",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            seed=602,
            metadata={"impostor_mode": "odd_even", "toi_label": "FP_vs_CP"},
        ),
        NeedleSpec(
            case_id="tess_decoy_03_kepler_giant_blend",
            category="look_alike_decoy",
            scenario_type="kepler_density_impostor",
            target_entity="TIC-149603524",
            decoy_entity="TIC-16740101",
            subtlety="high",
            description="Loud grazing binary decoy violates Kepler's 3rd Law stellar density by 18x; subtle real planet matches Gaia density.",
            expected_decisions={
                "confirmed_planet_tic": "TIC-149603524",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            seed=603,
            metadata={"impostor_mode": "density_mismatch", "toi_label": "FP_vs_CP"},
        ),
        NeedleSpec(
            case_id="tess_ud_01_crowded_aperture_pc",
            category="under_determined",
            scenario_type="crowded_field_candidate",
            target_entity="TIC-349827430",
            missing_signal="ccd_centroid_lock",
            subtlety="medium",
            description="Planetary candidate passes transit SNR, odd/even, and Keplerian density, but CROWDSAP=0.61 blends 3 stars (must remain UNCERTIFIED).",
            expected_decisions={
                "confirmed_planet_tic": "none_verified",
                "recommended_followup": "request_spectroscopy",
                "sufficiently_verified": "no",
            },
            seed=701,
            metadata={"impostor_mode": "crowded_pc", "toi_label": "PC"},
        ),
        NeedleSpec(
            case_id="tess_ud_02_low_pixel_snr_pc",
            category="under_determined",
            scenario_type="faint_stamp_candidate",
            target_entity="TIC-260304296",
            missing_signal="ccd_centroid_lock",
            subtlety="high",
            description="Clean folded light-curve dip & stellar density match, but per-pixel stamp SNR is too weak to certify spatial centroid lock.",
            expected_decisions={
                "confirmed_planet_tic": "none_verified",
                "recommended_followup": "request_spectroscopy",
                "sufficiently_verified": "no",
            },
            seed=702,
            metadata={"impostor_mode": "low_pix_snr_pc", "toi_label": "PC"},
        ),
        NeedleSpec(
            case_id="tess_ud_03_ambiguous_blend_apc",
            category="under_determined",
            scenario_type="ambiguous_pc",
            target_entity="TIC-278866211",
            missing_signal="ccd_centroid_lock",
            subtlety="medium",
            description="Ambiguous planetary candidate in crowded TESS field requiring follow-up radial velocity / high-res imaging.",
            expected_decisions={
                "confirmed_planet_tic": "none_verified",
                "recommended_followup": "request_spectroscopy",
                "sufficiently_verified": "no",
            },
            seed=703,
            metadata={"impostor_mode": "crowded_pc", "toi_label": "APC"},
        ),
    ]


class ExoplanetTriageDomain:
    """NASA TESS Exoplanet Vetting (`Domain` protocol implementation)."""

    name: str = "nasa_tess_exoplanet_triage"

    def __init__(self, staged_dir: str | Path | None = None) -> None:
        self._staged_dir = Path(staged_dir) if staged_dir else DEFAULT_STAGED_DIR
        manifest_file = self._staged_dir / "cases_manifest.json"
        if manifest_file.is_file():
            raw = json.loads(manifest_file.read_text())
            self._using_real_nasa_data = True
            self._candidate_tics: tuple[str, ...] = tuple(raw["candidate_tics"])
            self._menu: list[NeedleSpec] = [
                NeedleSpec.model_validate(item) for item in raw["cases"]
            ]
            self._nasa_truth_lookup: dict[str, dict[str, Any]] = raw.get("nasa_archive_truth", {})
        else:
            self._using_real_nasa_data = False
            self._candidate_tics = FALLBACK_CANDIDATE_TICS
            self._menu = _build_fallback_menu()
            self._nasa_truth_lookup = {}

    @property
    def is_using_real_nasa_data(self) -> bool:
        """True when backed by real staged NASA TESS SPOC FITS / Parquet observations."""
        return self._using_real_nasa_data

    def reveal_nasa_archive_truth(self, tic_id: str) -> dict[str, Any] | None:
        """Return official NASA Exoplanet Archive metadata for stage reveal (never in Parquet)."""
        return self._nasa_truth_lookup.get(tic_id)

    def create_backend(self, workspace_dir: str) -> ExecutionBackend:
        return DuckDBBackend(workspace_dir)

    def pivotal_fields(self) -> list[str]:
        return list(EXOPLANET_PIVOTAL_FIELDS)

    def required_signal_fields(self) -> tuple[str, ...]:
        return EXOPLANET_REQUIRED_SIGNALS

    def extract_field_patches_from_note(self, note: Note) -> dict[str, Any] | None:
        return extract_exoplanet_note_fields(note)

    def needle_menu(self) -> list[NeedleSpec]:
        return [n.model_copy(deep=True) for n in self._menu]

    def generate(self, out_dir: str, size_gb: float = 0.05, seed: int = 42) -> None:
        """Materialize background TESS sector haystack into partitioned Parquet shards."""
        root = Path(out_dir)
        for tbl in ("lightcurves", "stellar_catalog", "pixel_stamps"):
            tdir = root / tbl
            tdir.mkdir(parents=True, exist_ok=True)
            for existing in tdir.glob("*.parquet"):
                existing.unlink()

        bg_src = self._staged_dir / "background"
        if self._using_real_nasa_data and bg_src.is_dir():
            for tbl in ("lightcurves", "stellar_catalog", "pixel_stamps"):
                for src_pq in sorted((bg_src / tbl).glob("*.parquet")):
                    shutil.copy2(src_pq, root / tbl / src_pq.name)
            return

        # Offline fallback generator when staged NASA data directory is absent
        rng = np.random.default_rng(_stable_seed(seed, "tess_bg"))
        n_stars = 40
        n_cad = 1200
        lc_tables: list[pa.Table] = []
        cat_rows: dict[str, list[Any]] = {f.name: [] for f in STELLAR_CATALOG_SCHEMA}
        for s_i in range(n_stars):
            tic = f"TIC-9000{s_i:04d}"
            btjd = np.linspace(1325.0, 1352.0, n_cad)
            err = np.full(n_cad, 220.0)
            rel = rng.normal(0.0, 220.0, size=n_cad)
            lc_tables.append(
                pa.table(
                    {
                        "tic_id": pa.array([tic] * n_cad, type=pa.string()),
                        "sector": pa.array(np.full(n_cad, 1, dtype=np.int32), type=pa.int32()),
                        "cadence_no": pa.array(np.arange(1, n_cad + 1, dtype=np.int32), type=pa.int32()),
                        "btjd": pa.array(btjd, type=pa.float64()),
                        "pdcsap_flux": pa.array(10000.0 * (1.0 + rel * 1e-6), type=pa.float64()),
                        "rel_flux_ppm": pa.array(rel, type=pa.float64()),
                        "flux_err_ppm": pa.array(err, type=pa.float64()),
                        "mom_centr1": pa.array(np.full(n_cad, 5.0), type=pa.float64()),
                        "mom_centr2": pa.array(np.full(n_cad, 5.0), type=pa.float64()),
                        "quality_flag": pa.array(np.zeros(n_cad, dtype=np.int32), type=pa.int32()),
                    },
                    schema=LIGHTCURVES_SCHEMA,
                )
            )
            cat_rows["tic_id"].append(tic)
            cat_rows["sector"].append(1)
            cat_rows["tmag"].append(round(float(rng.uniform(9.5, 12.8)), 3))
            cat_rows["teff_k"].append(round(float(rng.uniform(4200, 6300)), 1))
            cat_rows["stellar_radius_rsun"].append(1.0)
            cat_rows["stellar_mass_msun"].append(1.0)
            cat_rows["stellar_density_gcc"].append(1.408)
            cat_rows["crowding_metric"].append(0.95)
            cat_rows["flux_fraction"].append(0.88)
            cat_rows["ra_deg"].append(120.0 + s_i * 0.1)
            cat_rows["dec_deg"].append(-45.0 + s_i * 0.1)
            cat_rows["candidate_period_days"].append(3.5)
            cat_rows["candidate_t0_btjd"].append(1326.0)
            cat_rows["candidate_duration_hours"].append(2.5)

        pq.write_table(pa.concat_tables(lc_tables), root / "lightcurves" / "bg_part_0000.parquet")
        pq.write_table(
            pa.table({k: pa.array(v, type=STELLAR_CATALOG_SCHEMA.field(k).type) for k, v in cat_rows.items()}, schema=STELLAR_CATALOG_SCHEMA),
            root / "stellar_catalog" / "bg_part_0000.parquet",
        )
        pq.write_table(
            pa.table({f.name: pa.array([], type=f.type) for f in PIXEL_STAMPS_SCHEMA}, schema=PIXEL_STAMPS_SCHEMA),
            root / "pixel_stamps" / "bg_part_0000.parquet",
        )

    def _synthesize_fallback_star(
        self,
        tic_id: str,
        *,
        mode: str,
        seed: int,
    ) -> tuple[pa.Table, pa.Table, pa.Table]:
        rng = np.random.default_rng(_stable_seed(seed, tic_id))
        n_cad = 3200
        btjd = np.linspace(1325.0, 1352.0, n_cad)
        p_days = 3.2
        t0 = 1326.1
        dur_h = 2.6
        epoch = np.round((btjd - t0) / p_days).astype(np.int64)
        phase_h = ((btjd - t0) - epoch * p_days) * 24.0
        in_tr = np.abs(phase_h) <= (0.42 * dur_h)
        is_odd = (np.abs(epoch) % 2) == 1

        err_ppm = np.full(n_cad, 180.0)
        rel_ppm = rng.normal(0.0, 180.0, size=n_cad)

        depth = 1600.0 if mode == "planet" else 5200.0
        if mode == "odd_even":
            rel_ppm[in_tr & is_odd] -= 6800.0
            rel_ppm[in_tr & (~is_odd)] -= 2400.0
        else:
            rel_ppm[in_tr] -= depth

        rho_gaia = 1.408 if mode != "density_mismatch" else 0.035
        crowding = 0.61 if mode == "crowded_pc" else 0.96

        lc_t = pa.table(
            {
                "tic_id": pa.array([tic_id] * n_cad, type=pa.string()),
                "sector": pa.array(np.full(n_cad, 1, dtype=np.int32), type=pa.int32()),
                "cadence_no": pa.array(np.arange(1, n_cad + 1, dtype=np.int32), type=pa.int32()),
                "btjd": pa.array(btjd, type=pa.float64()),
                "pdcsap_flux": pa.array(12000.0 * (1.0 + rel_ppm * 1e-6), type=pa.float64()),
                "rel_flux_ppm": pa.array(rel_ppm, type=pa.float64()),
                "flux_err_ppm": pa.array(err_ppm, type=pa.float64()),
                "mom_centr1": pa.array(np.full(n_cad, 5.0), type=pa.float64()),
                "mom_centr2": pa.array(np.full(n_cad, 5.0), type=pa.float64()),
                "quality_flag": pa.array(np.zeros(n_cad, dtype=np.int32), type=pa.int32()),
            },
            schema=LIGHTCURVES_SCHEMA,
        )
        cat_t = pa.table(
            {
                "tic_id": pa.array([tic_id], type=pa.string()),
                "sector": pa.array([1], type=pa.int32()),
                "tmag": pa.array([10.4], type=pa.float64()),
                "teff_k": pa.array([5620.0], type=pa.float64()),
                "stellar_radius_rsun": pa.array([1.0], type=pa.float64()),
                "stellar_mass_msun": pa.array([1.0], type=pa.float64()),
                "stellar_density_gcc": pa.array([rho_gaia], type=pa.float64()),
                "crowding_metric": pa.array([crowding], type=pa.float64()),
                "flux_fraction": pa.array([0.89], type=pa.float64()),
                "ra_deg": pa.array([112.5], type=pa.float64()),
                "dec_deg": pa.array([-52.3], type=pa.float64()),
                "candidate_period_days": pa.array([p_days], type=pa.float64()),
                "candidate_t0_btjd": pa.array([t0], type=pa.float64()),
                "candidate_duration_hours": pa.array([dur_h], type=pa.float64()),
            },
            schema=STELLAR_CATALOG_SCHEMA,
        )

        # 11x11 pixel stamp
        shift_r, shift_c = (2.0, 1.5) if mode == "centroid_shift" else (0.04, -0.03)
        peak_amp = 2.8 if mode == "low_pix_snr_pc" else 18.0
        prows: dict[str, list[Any]] = {f.name: [] for f in PIXEL_STAMPS_SCHEMA}
        for r_i in range(11):
            for c_i in range(11):
                d2_target = (r_i - 5.0) ** 2 + (c_i - 5.0) ** 2
                d2_cent = (r_i - (5.0 + shift_r)) ** 2 + (c_i - (5.0 + shift_c)) ** 2
                mout = 2500.0 * math.exp(-d2_target / 1.8) + 20.0
                dsnr = peak_amp * math.exp(-d2_cent / 1.4)
                dflux = dsnr * 3.0
                amask = 3 if d2_target <= 2.3 else 1
                prows["tic_id"].append(tic_id)
                prows["sector"].append(1)
                prows["row_idx"].append(r_i)
                prows["col_idx"].append(c_i)
                prows["mean_out_flux"].append(mout)
                prows["mean_in_flux"].append(mout - dflux)
                prows["diff_flux"].append(dflux)
                prows["diff_snr"].append(dsnr)
                prows["aperture_mask"].append(amask)
                prows["target_row"].append(5.0)
                prows["target_col"].append(5.0)

        pix_t = pa.table(
            {k: pa.array(v, type=PIXEL_STAMPS_SCHEMA.field(k).type) for k, v in prows.items()},
            schema=PIXEL_STAMPS_SCHEMA,
        )
        return lc_t, cat_t, pix_t

    def plant(self, out_dir: str, needle: NeedleSpec) -> GroundTruth:
        """Append real NASA target Parquet shard(s) for `needle` into `out_dir`."""
        root = Path(out_dir)
        for tbl in ("lightcurves", "stellar_catalog", "pixel_stamps"):
            (root / tbl).mkdir(parents=True, exist_ok=True)

        stars_to_plant: list[tuple[str, str]] = []
        if needle.category == "multi_clue_join":
            stars_to_plant.append((needle.target_entity, "planet"))
        elif needle.category == "look_alike_decoy":
            assert needle.decoy_entity is not None
            impostor_mode = str(needle.metadata.get("impostor_mode", "centroid_shift"))
            stars_to_plant.append((needle.decoy_entity, impostor_mode))
            stars_to_plant.append((needle.target_entity, "planet"))
        elif needle.category == "under_determined":
            impostor_mode = str(needle.metadata.get("impostor_mode", "crowded_pc"))
            stars_to_plant.append((needle.target_entity, impostor_mode))
        else:
            raise ValueError(f"Unknown case category: {needle.category}")

        lc_parts: list[pa.Table] = []
        cat_parts: list[pa.Table] = []
        pix_parts: list[pa.Table] = []

        for tic_id, fallback_mode in stars_to_plant:
            staged_star_dir = self._staged_dir / "targets" / tic_id
            if self._using_real_nasa_data and staged_star_dir.is_dir():
                lc_parts.append(pq.read_table(staged_star_dir / "lightcurves.parquet"))
                cat_parts.append(pq.read_table(staged_star_dir / "stellar_catalog.parquet"))
                pix_parts.append(pq.read_table(staged_star_dir / "pixel_stamps.parquet"))
            else:
                lc_t, cat_t, pix_t = self._synthesize_fallback_star(
                    tic_id, mode=fallback_mode, seed=needle.seed
                )
                lc_parts.append(lc_t)
                cat_parts.append(cat_t)
                pix_parts.append(pix_t)

        planted_files: list[str] = []
        for tbl_name, parts in (
            ("lightcurves", lc_parts),
            ("stellar_catalog", cat_parts),
            ("pixel_stamps", pix_parts),
        ):
            dest = root / tbl_name / f"planted_{needle.case_id}.parquet"
            pq.write_table(pa.concat_tables(parts), dest)
            planted_files.append(str(dest))

        is_verified = needle.category != "under_determined"
        primary_target = needle.target_entity if is_verified else None
        expected_decisions = dict(needle.expected_decisions)
        if not expected_decisions:
            expected_decisions = {
                "confirmed_planet_tic": needle.target_entity if is_verified else "none_verified",
                "recommended_followup": (
                    "trigger_flagship_slew" if is_verified else "request_spectroscopy"
                ),
                "sufficiently_verified": "yes" if is_verified else "no",
            }

        return GroundTruth(
            case_id=needle.case_id,
            category=needle.category,
            primary_target=primary_target,
            expected_decisions=expected_decisions,
            is_sufficiently_verified=is_verified,
            expected_status="CERTIFIED" if is_verified else "UNCERTIFIED",
            pivotal_fields=list(EXOPLANET_PIVOTAL_FIELDS),
            decoy_target=needle.decoy_entity,
            planted_artifacts=planted_files,
            metadata=dict(needle.metadata),
        )

    def environment_description(self) -> str:
        return (
            "NASA TESS Space Telescope DuckDB SQL views (auto-registered; query by view name directly):\n"
            "1. lightcurves:\n"
            "   - tic_id (VARCHAR, e.g. 'TIC-261136679'), sector (INTEGER), cadence_no (INTEGER), "
            "btjd (DOUBLE, Barycentric TESS Julian Date), pdcsap_flux (DOUBLE, e-/s), "
            "rel_flux_ppm (DOUBLE, continuum-normalized relative flux in ppm), "
            "flux_err_ppm (DOUBLE, 1-sigma flux uncertainty in ppm), "
            "mom_centr1 (DOUBLE), mom_centr2 (DOUBLE), quality_flag (INTEGER, 0 = clean SPOC cadence).\n"
            "2. stellar_catalog:\n"
            "   - tic_id (VARCHAR), sector (INTEGER), tmag (DOUBLE), teff_k (DOUBLE), "
            "stellar_radius_rsun (DOUBLE), stellar_mass_msun (DOUBLE), "
            "stellar_density_gcc (DOUBLE, Gaia/TIC host density in g/cm^3), "
            "crowding_metric (DOUBLE, NASA SPOC CROWDSAP in [0,1]: fraction of aperture flux from target star), "
            "flux_fraction (DOUBLE), ra_deg (DOUBLE), dec_deg (DOUBLE), "
            "candidate_period_days (DOUBLE, BLS trial transit-search period P in days), "
            "candidate_t0_btjd (DOUBLE), candidate_duration_hours (DOUBLE).\n"
            "3. pixel_stamps:\n"
            "   - 11x11 NASA TESS Target Pixel File (TPF) transit difference image per candidate star:\n"
            "   - tic_id (VARCHAR), sector (INTEGER), row_idx (INTEGER), col_idx (INTEGER), "
            "mean_out_flux (DOUBLE), mean_in_flux (DOUBLE), diff_flux (DOUBLE, out - in transit), "
            "diff_snr (DOUBLE, per-pixel transit dimming SNR), aperture_mask (INTEGER, bitmask where (aperture_mask & 2) != 0 is SPOC optimal target aperture), "
            "target_row (DOUBLE), target_col (DOUBLE, catalog target star pixel coordinates).\n"
            "Physical Vetting Rules (a CERTIFIED exoplanet MUST pass ALL 4 independent checks on the same tic_id):\n"
            "  (1) `periodic_transit_snr`: >=2 transits, folded depth >= 150 ppm, transit SNR >= 7.0 (note: loud eclipsing binary impostors ALSO pass this!).\n"
            "  (2) `odd_even_eclipse_clean`: odd vs. even transit depth difference < 3.0 sigma AND phase-0.5 secondary occultation SNR < 4.5.\n"
            "  (3) `keplerian_density_match`: Kepler's 3rd Law transit-implied stellar density ratio (rho_transit / stellar_density_gcc) in [0.18, 5.5].\n"
            "  (4) `ccd_centroid_lock`: diff_snr^2-weighted pixel centroid offset <= 0.45 pixels from (target_row, target_col), peak pixel diff_snr >= 5.5, AND crowding_metric >= 0.80."
        )

    def magnet_rules(self) -> str:
        return (
            "1. Reference DuckDB table view names directly (`lightcurves`, `stellar_catalog`, `pixel_stamps`). "
            "NEVER call `read_parquet(...)` and NEVER include file paths.\n"
            "2. Emit single read-only `SELECT` or `WITH ... SELECT` queries only (no semicolons, no DDL/DML).\n"
            "3. Bound result sets with `GROUP BY` / `ORDER BY` / `LIMIT <= 50`."
        )

    def hypothesis_families(self) -> list[str]:
        return [
            "periodic_transit_photometry_scan",
            "odd_even_and_secondary_eclipse_vetting",
            "keplerian_orbital_density_consistency",
            "ccd_pixel_centroid_impostor_check",
            "four_way_exoplanet_confirmation",
        ]

    def search_questions(self) -> list[SearchQuestion]:
        tic_options = [*self._candidate_tics, "none_verified"]
        req_str = (
            "`periodic_transit_snr`, `odd_even_eclipse_clean`, "
            "`keplerian_density_match`, `ccd_centroid_lock`"
        )
        tic_criteria: dict[str, str] = {
            "none_verified": (
                f"Select if ANY of the four required signal fields ({req_str}) is missing, "
                "has status != 'certified', or has value in ('unknown', 'none'), OR if no single "
                "TIC star appears across ALL FOUR required signal fields."
            )
        }
        for tic in self._candidate_tics:
            tic_criteria[tic] = (
                f"Select when all four required signal fields ({req_str}) have "
                f"status='certified' and '{tic}' appears in ALL FOUR of them (each field may hold "
                "either a single TIC string or a list of multiple TIC strings when a partial decoy "
                "co-exists with the confirmed planet; corroborate with `confirmed_exoplanet_target` "
                "when status='certified')."
            )

        followup_criteria = {
            "trigger_flagship_slew": (
                f"Select WHENEVER a single TIC star appears across ALL FOUR `status='certified'` "
                f"required signal fields ({req_str}) (whether as a single string or inside a "
                "multi-candidate list) and/or `confirmed_exoplanet_target` has status='certified' "
                "with that TIC ID, confirming a genuine exoplanet ready for flagship JWST/Keck "
                "characterization."
            ),
            "request_spectroscopy": (
                f"Select when NO star is certified across all four required signal fields ({req_str}) "
                "—for example when `ccd_centroid_lock` or another required signal is 'none', "
                "'unknown', or `status != 'certified'` due to aperture crowding or unconfirmed physics."
            ),
            "reject_all_candidates": (
                "Select ONLY when `periodic_transit_snr` has status='certified' and value='none' "
                "(zero transit dips detected anywhere in the sector)."
            ),
        }

        return [
            SearchQuestion(
                question_id="confirmed_planet_tic",
                question="Which TESS candidate star in this sector is a deterministically confirmed exoplanet?",
                options=tic_options,
                question_type="multiple_choice",
                instructions=(
                    "Evaluate strictly from the board's `fields` (`value` and `status`). "
                    f"The four required deterministic signal fields are {req_str}. "
                    "Individual signal fields may hold either a single TIC string OR a list of "
                    "multiple TIC strings when a partial 2-of-4 or 3-of-4 cosmic impostor co-exists "
                    "in the sector alongside a true 4-of-4 confirmed planet. Whenever a single TIC "
                    "star appears in ALL FOUR `status='certified'` required signal fields (and "
                    "matches `confirmed_exoplanet_target` with status='certified'), you MUST select "
                    "that exact TIC ID; select 'none_verified' only when no single TIC star appears "
                    "in all four certified signal fields."
                ),
                option_criteria=tic_criteria,
            ),
            SearchQuestion(
                question_id="recommended_followup",
                question="What observatory follow-up action is warranted?",
                options=["trigger_flagship_slew", "request_spectroscopy", "reject_all_candidates"],
                question_type="multiple_choice",
                instructions=(
                    "Select 'trigger_flagship_slew' WHENEVER a single TIC candidate appears across "
                    f"all four `status='certified'` required signal fields ({req_str}) (including "
                    "when some fields contain a multi-TIC list alongside a decoy and "
                    "`confirmed_exoplanet_target` is certified); select 'request_spectroscopy' only "
                    "when no TIC candidate achieves full 4-way certification."
                ),
                option_criteria=followup_criteria,
            ),
            SearchQuestion(
                question_id="sufficiently_verified",
                question="Is an exoplanet candidate deterministically verified across all four photometric, orbital, and CCD spatial checks?",
                options=["yes", "no"],
                question_type="yes_no",
                instructions=(
                    f"Evaluate strictly from {req_str}. Answer true whenever at least "
                    "one TIC star appears across ALL FOUR required signal fields (either as a single "
                    "string or inside a list of candidates) and every one of those four fields has "
                    "`status='certified'` (matching `confirmed_exoplanet_target` when certified). "
                    "Answer false whenever ANY of those four fields is missing, has "
                    "status != 'certified', or has value in ('none', 'unknown'), or when no single "
                    "TIC star is in the intersection of all four."
                ),
            ),
        ]

    def resolve_answers(
        self,
        board: InvestigationBoard,
        questions: Sequence[SearchQuestion],
        primary_target: str | None,
    ) -> dict[str, str]:
        """Deterministically resolve observatory follow-up action from board state."""
        del questions
        if primary_target is not None:
            return {"recommended_followup": "trigger_flagship_slew", "sufficiently_verified": "yes"}
        snr_fld = board.fields.get("periodic_transit_snr")
        if snr_fld is not None and snr_fld.status == "certified" and snr_fld.value == "none":
            return {"recommended_followup": "reject_all_candidates", "sufficiently_verified": "no"}
        return {"recommended_followup": "request_spectroscopy", "sufficiently_verified": "no"}

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        registry.register(
            "verify_periodic_transit_snr",
            verify_periodic_transit_snr,
            description="Certifies TESS candidates with >=2 clean transits, folded depth >= 150 ppm, and transit SNR >= 7.0.",
            trusted=True,
        )
        registry.register(
            "verify_odd_even_eclipse_symmetry",
            verify_odd_even_eclipse_symmetry,
            description="Certifies transit candidates with symmetric odd/even depths (<3.0 sigma) and no secondary eclipse (<4.5 sigma).",
            trusted=True,
        )
        registry.register(
            "verify_keplerian_density_consistency",
            verify_keplerian_density_consistency,
            description="Certifies transit candidates whose Kepler's 3rd Law orbital stellar density matches Gaia/TIC host density within [0.18x, 5.5x].",
            trusted=True,
        )
        registry.register(
            "verify_ccd_centroid_lock",
            verify_ccd_centroid_lock,
            description="Certifies transit candidates whose 11x11 TESS CCD difference centroid locks within <=0.45 px of target star with CROWDSAP >= 0.80.",
            trusted=True,
        )
        registry.register(
            "verify_exoplanet_confirmation",
            verify_exoplanet_confirmation,
            description="Compound 4-way NASA TESS confirmation verifier joining transit SNR, odd/even symmetry, Keplerian density, and CCD centroid lock, and emits 2x2 diagnostic PNGs.",
            trusted=True,
        )

    def score(self, board: dict[str, Any], truth: GroundTruth) -> dict[str, Any]:
        """Evaluate final board answer, decoy resistance, certification discipline, and narrowing."""
        conc = board.get("conclusion") or board.get("decision") or {}
        concs_map = board.get("conclusions") or board.get("decisions") or {}

        tic_conc = concs_map.get("confirmed_planet_tic", conc)
        act_conc = concs_map.get("recommended_followup", {})
        ver_conc = concs_map.get("sufficiently_verified", {})

        predicted_tic = tic_conc.get("answer", conc.get("answer", "unknown"))
        predicted_action = act_conc.get("answer", "request_spectroscopy")
        predicted_verified_raw = ver_conc.get("answer")

        confidence = float(conc.get("confidence", 0.0))
        verified_flag = conc.get("verified")
        meta = conc.get("metadata") or {}
        uncertified_share = float(
            meta.get(
                "uncertified_influence_share",
                conc.get("uncertified_influence_share", 0.0 if verified_flag else 1.0),
            )
        )
        unknown_voi_max = float(
            meta.get(
                "unknown_voi_max",
                conc.get("unknown_voi_max", 0.0 if verified_flag else 1.0),
            )
        )

        if verified_flag is not None:
            board_certified = (
                bool(verified_flag)
                and confidence >= 0.85
                and predicted_tic not in (None, "", "unknown", "none_verified")
                and (predicted_verified_raw in (None, "yes"))
            )
        else:
            board_certified = (
                confidence >= 0.85
                and uncertified_share <= 0.15
                and unknown_voi_max < 0.20
                and predicted_tic not in (None, "", "unknown", "none_verified")
                and (predicted_verified_raw in (None, "yes"))
            )

        if truth.is_sufficiently_verified:
            target_correct = predicted_tic == truth.primary_target
        else:
            target_correct = (
                not board_certified
                or predicted_tic in (None, "none_verified", "unknown")
            )

        expected_action = truth.expected_decisions.get(
            "recommended_followup", "request_spectroscopy"
        )
        action_correct = predicted_action == expected_action
        certification_correct = board_certified == truth.is_sufficiently_verified
        false_certainty = (not truth.is_sufficiently_verified) and board_certified
        decoy_fallen_for = (
            truth.decoy_target is not None and predicted_tic == truth.decoy_target
        )
        decoy_avoided = (
            None if truth.decoy_target is None else (not decoy_fallen_for)
        )

        fields = board.get("fields", {})
        pivotal_set = set(truth.pivotal_fields)
        pivotal_priorities = [
            float(f.get("priority", f.get("influence", 0.0)))
            for fname, f in fields.items()
            if fname in pivotal_set
        ]
        distractor_priorities = [
            float(f.get("priority", f.get("influence", 0.0)))
            for fname, f in fields.items()
            if fname not in pivotal_set and f.get("status") != "unknown"
        ]
        pivotal_priority_mean = (
            float(np.mean(pivotal_priorities)) if pivotal_priorities else None
        )
        distractor_priority_mean = (
            float(np.mean(distractor_priorities)) if distractor_priorities else None
        )

        total_candidates = len(self._candidate_tics)
        if predicted_tic in self._candidate_tics:
            candidate_narrowing = round(1.0 - (1.0 / max(1, total_candidates)), 4)
        else:
            candidate_narrowing = 0.0

        overall_pass = (
            target_correct
            and certification_correct
            and not decoy_fallen_for
            and not false_certainty
        )

        return {
            "case_id": truth.case_id,
            "category": truth.category,
            "expected_status": truth.expected_status,
            "predicted_root_cause": predicted_tic,
            "predicted_target": predicted_tic,
            "predicted_action": predicted_action,
            "confidence": confidence,
            "certified_field_ratio": float(conc.get("certified_field_ratio", 0.0)),
            "board_certified": board_certified,
            "root_cause_correct": target_correct,
            "target_correct": target_correct,
            "action_correct": action_correct,
            "certification_correct": certification_correct,
            "false_certainty": false_certainty,
            "decoy_fallen_for": decoy_fallen_for,
            "decoy_avoided": decoy_avoided,
            "pivotal_priority_mean": pivotal_priority_mean,
            "distractor_priority_mean": distractor_priority_mean,
            "candidate_narrowing": candidate_narrowing,
            "overall_pass": overall_pass,
        }


# ---------------------------------------------------------------------------
# Deterministic TRUSTED Physics & Spatial Pixel Verifiers (Zero LLM Calls)
# ---------------------------------------------------------------------------


def _query_all_candidate_metrics(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    tic_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Compute exact photometric, odd/even, Keplerian density, and CCD centroid metrics in DuckDB."""
    lc_glob = _parquet_glob(data_dir, "lightcurves")
    cat_glob = _parquet_glob(data_dir, "stellar_catalog")
    pix_glob = _parquet_glob(data_dir, "pixel_stamps")

    where_tic = "AND s.tic_id = ?" if tic_filter else ""
    q_params: list[Any] = [tic_filter] if tic_filter else []

    sql = f"""
        WITH folded AS (
            SELECT
                l.tic_id,
                s.sector,
                s.tmag,
                s.teff_k,
                s.stellar_radius_rsun,
                s.stellar_density_gcc AS rho_gaia,
                s.crowding_metric,
                s.candidate_period_days AS p_days,
                s.candidate_t0_btjd AS t0,
                s.candidate_duration_hours AS dur_h,
                l.btjd,
                l.rel_flux_ppm,
                l.flux_err_ppm,
                CAST(ROUND((l.btjd - s.candidate_t0_btjd) / s.candidate_period_days) AS BIGINT) AS epoch_idx,
                ((l.btjd - s.candidate_t0_btjd)
                 - ROUND((l.btjd - s.candidate_t0_btjd) / s.candidate_period_days) * s.candidate_period_days
                ) * 24.0 AS phase_h,
                ((((l.btjd - (s.candidate_t0_btjd + 0.5 * s.candidate_period_days))
                   % s.candidate_period_days) + 1.5 * s.candidate_period_days)
                 % s.candidate_period_days - 0.5 * s.candidate_period_days
                ) * 24.0 AS sec_phase_h
            FROM read_parquet('{lc_glob}') l
            INNER JOIN read_parquet('{cat_glob}') s ON l.tic_id = s.tic_id
            WHERE l.quality_flag = 0
              AND l.rel_flux_ppm IS NOT NULL
              AND s.candidate_period_days IS NOT NULL
              {where_tic}
        ),
        lc_stats AS (
            SELECT
                tic_id,
                ANY_VALUE(sector) AS sector,
                ANY_VALUE(tmag) AS tmag,
                ANY_VALUE(p_days) AS p_days,
                ANY_VALUE(dur_h) AS dur_h,
                ANY_VALUE(rho_gaia) AS rho_gaia,
                ANY_VALUE(crowding_metric) AS crowding_metric,
                COUNT(DISTINCT epoch_idx) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h) AS n_transits,
                COUNT_IF(ABS(phase_h) <= 0.425 * dur_h) AS n_in,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) BETWEEN 0.75 * dur_h AND 2.0 * dur_h) AS out_med,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h) AS in_med,
                COALESCE(MEDIAN(flux_err_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h), 250.0) AS in_err,
                COUNT_IF(ABS(phase_h) <= 0.425 * dur_h AND ABS(epoch_idx) % 2 = 1) AS n_odd,
                COUNT_IF(ABS(phase_h) <= 0.425 * dur_h AND ABS(epoch_idx) % 2 = 0) AS n_even,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h AND ABS(epoch_idx) % 2 = 1) AS odd_in_med,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h AND ABS(epoch_idx) % 2 = 0) AS even_in_med,
                COUNT_IF(ABS(sec_phase_h) <= 0.425 * dur_h) AS n_sec_in,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(sec_phase_h) BETWEEN 0.75 * dur_h AND 2.0 * dur_h) AS sec_out_med,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(sec_phase_h) <= 0.425 * dur_h) AS sec_in_med
            FROM folded
            GROUP BY tic_id
        ),
        pix_stats AS (
            SELECT
                tic_id,
                SUM(row_idx * POW(GREATEST(0.0, diff_snr), 2)) / NULLIF(SUM(POW(GREATEST(0.0, diff_snr), 2)), 0) AS cent_r,
                SUM(col_idx * POW(GREATEST(0.0, diff_snr), 2)) / NULLIF(SUM(POW(GREATEST(0.0, diff_snr), 2)), 0) AS cent_c,
                AVG(target_row) AS t_row,
                AVG(target_col) AS t_col,
                MAX(diff_snr) AS peak_pix_snr
            FROM read_parquet('{pix_glob}')
            GROUP BY tic_id
        )
        SELECT
            ls.tic_id,
            ls.sector,
            ls.tmag,
            ls.p_days,
            ls.dur_h,
            ls.rho_gaia,
            ls.crowding_metric,
            ls.n_transits,
            ls.n_in,
            ls.out_med,
            ls.in_med,
            ls.in_err,
            ls.n_odd,
            ls.n_even,
            ls.odd_in_med,
            ls.even_in_med,
            ls.n_sec_in,
            ls.sec_out_med,
            ls.sec_in_med,
            ps.cent_r,
            ps.cent_c,
            ps.t_row,
            ps.t_col,
            ps.peak_pix_snr
        FROM lc_stats ls
        LEFT JOIN pix_stats ps ON ls.tic_id = ps.tic_id
        ORDER BY ls.tic_id ASC
    """
    rows = con.execute(sql, q_params).fetchall()
    out: list[dict[str, Any]] = []

    for r in rows:
        (
            tic_id,
            sector,
            tmag,
            p_days,
            dur_h,
            rho_gaia,
            crowding,
            n_transits,
            n_in,
            out_med,
            in_med,
            in_err,
            n_odd,
            n_even,
            odd_in_med,
            even_in_med,
            n_sec_in,
            sec_out_med,
            sec_in_med,
            cent_r,
            cent_c,
            t_row,
            t_col,
            peak_pix_snr,
        ) = r

        if int(n_in or 0) < 4 or out_med is None or in_med is None:
            continue

        depth_ppm = max(0.0, float(out_med) - float(in_med))
        in_err_f = float(in_err or 250.0)
        transit_snr = depth_ppm / max(15.0, in_err_f / math.sqrt(max(1, int(n_in))))

        if int(n_odd or 0) >= 2 and int(n_even or 0) >= 2 and odd_in_med is not None and even_in_med is not None:
            oe_diff = abs(float(odd_in_med) - float(even_in_med))
            oe_err = max(40.0, in_err_f / math.sqrt(max(1, min(int(n_odd), int(n_even)))))
            oe_sigma = oe_diff / oe_err
        else:
            oe_sigma = 99.0

        if int(n_sec_in or 0) >= 3 and sec_out_med is not None and sec_in_med is not None:
            sec_depth = max(0.0, float(sec_out_med) - float(sec_in_med))
            sec_snr = sec_depth / max(20.0, in_err_f / math.sqrt(max(1, int(n_sec_in))))
        else:
            sec_snr = 0.0

        k = math.sqrt(min(0.25, max(1e-5, depth_ppm * 1e-6)))
        sin_term = math.sin(min(math.pi / 2.1, math.pi * (float(dur_h) / 24.0) / float(p_days)))
        a_r = max(1.2, (1.0 + k) / max(1e-4, sin_term))
        rho_tr = (3.0 * math.pi / (6.6743e-8 * ((float(p_days) * 86400.0) ** 2))) * (a_r ** 3)
        density_ratio = rho_tr / max(1e-4, float(rho_gaia or 1.408))

        if cent_r is not None and t_row is not None:
            cent_offset_px = math.hypot(float(cent_r) - float(t_row), float(cent_c) - float(t_col))
            peak_snr_val = float(peak_pix_snr or 0.0)
        else:
            cent_offset_px = 99.0
            peak_snr_val = 0.0

        crowd_val = float(crowding if crowding is not None else 0.0)

        out.append(
            {
                "tic_id": str(tic_id),
                "sector": int(sector),
                "tmag": round(float(tmag), 3),
                "period_days": round(float(p_days), 5),
                "duration_hours": round(float(dur_h), 3),
                "n_transits": int(n_transits),
                "depth_ppm": round(depth_ppm, 1),
                "transit_snr": round(transit_snr, 2),
                "odd_even_sigma": round(oe_sigma, 3),
                "secondary_snr": round(sec_snr, 2),
                "rho_transit_gcc": round(rho_tr, 4),
                "rho_gaia_gcc": round(float(rho_gaia), 4),
                "density_ratio": round(density_ratio, 3),
                "centroid_offset_px": round(cent_offset_px, 3),
                "peak_pixel_snr": round(peak_snr_val, 2),
                "crowding_metric": round(crowd_val, 4),
            }
        )

    out.sort(key=lambda item: (-item["transit_snr"], item["tic_id"]))
    return out


def verify_periodic_transit_snr(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    params: dict[str, Any] | None = None,
) -> VerifierResult:
    params = params or {}
    min_snr = float(params.get("min_snr", 7.0))
    min_depth_ppm = float(params.get("min_depth_ppm", 150.0))
    min_transits = int(params.get("min_transits", 2))

    # Sector-wide evaluation ensures follow-up calls cannot overwrite sector candidate sets.
    all_cands = _query_all_candidate_metrics(con, data_dir, tic_filter=None)
    passing = [
        {
            "tic_id": c["tic_id"],
            "sector": c["sector"],
            "period_days": c["period_days"],
            "n_transits": c["n_transits"],
            "depth_ppm": c["depth_ppm"],
            "transit_snr": c["transit_snr"],
        }
        for c in all_cands
        if c["n_transits"] >= min_transits
        and c["transit_snr"] >= min_snr
        and c["depth_ppm"] >= min_depth_ppm
    ]
    passed = len(passing) > 0
    return VerifierResult(
        verifier_name="verify_periodic_transit_snr",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={"candidates": passing},
        row_count=len(passing),
        summary=(
            f"Detected {len(passing)} periodic transit candidate(s): {[c['tic_id'] for c in passing]}"
            if passed
            else "No periodic transit signal met SNR/depth threshold."
        ),
    )


def verify_odd_even_eclipse_symmetry(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    params: dict[str, Any] | None = None,
) -> VerifierResult:
    params = params or {}
    max_oe_sigma = float(params.get("max_odd_even_sigma", 3.0))
    max_sec_snr = float(params.get("max_secondary_snr", 4.5))

    all_cands = _query_all_candidate_metrics(con, data_dir, tic_filter=None)
    snr_cands = [
        c for c in all_cands if c["n_transits"] >= 2 and c["transit_snr"] >= 7.0 and c["depth_ppm"] >= 150.0
    ]
    passing = [
        {
            "tic_id": c["tic_id"],
            "odd_even_sigma": c["odd_even_sigma"],
            "secondary_snr": c["secondary_snr"],
            "transit_snr": c["transit_snr"],
        }
        for c in snr_cands
        if c["odd_even_sigma"] < max_oe_sigma and c["secondary_snr"] < max_sec_snr
    ]
    rejected = [
        {
            "tic_id": c["tic_id"],
            "odd_even_sigma": c["odd_even_sigma"],
            "secondary_snr": c["secondary_snr"],
        }
        for c in snr_cands
        if not (c["odd_even_sigma"] < max_oe_sigma and c["secondary_snr"] < max_sec_snr)
    ]
    passed = len(passing) > 0
    return VerifierResult(
        verifier_name="verify_odd_even_eclipse_symmetry",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={"candidates": passing, "rejected_eclipsing_binaries": rejected},
        row_count=len(passing),
        summary=(
            f"Symmetric odd/even transits verified for {[c['tic_id'] for c in passing]} "
            f"(pruned binary asymmetries: {[r['tic_id'] for r in rejected]})."
            if passed
            else "No candidate satisfied odd/even eclipse symmetry and secondary occultation limits."
        ),
    )


def verify_keplerian_density_consistency(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    params: dict[str, Any] | None = None,
) -> VerifierResult:
    params = params or {}
    min_ratio = float(params.get("min_density_ratio", 0.18))
    max_ratio = float(params.get("max_density_ratio", 5.5))

    all_cands = _query_all_candidate_metrics(con, data_dir, tic_filter=None)
    snr_cands = [
        c for c in all_cands if c["n_transits"] >= 2 and c["transit_snr"] >= 7.0 and c["depth_ppm"] >= 150.0
    ]
    passing = [
        {
            "tic_id": c["tic_id"],
            "rho_transit_gcc": c["rho_transit_gcc"],
            "rho_gaia_gcc": c["rho_gaia_gcc"],
            "density_ratio": c["density_ratio"],
        }
        for c in snr_cands
        if min_ratio <= c["density_ratio"] <= max_ratio
    ]
    rejected = [
        {
            "tic_id": c["tic_id"],
            "density_ratio": c["density_ratio"],
        }
        for c in snr_cands
        if not (min_ratio <= c["density_ratio"] <= max_ratio)
    ]
    passed = len(passing) > 0
    return VerifierResult(
        verifier_name="verify_keplerian_density_consistency",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={"candidates": passing, "rejected_density_mismatches": rejected},
        row_count=len(passing),
        summary=(
            f"Kepler's 3rd Law stellar density match verified for {[c['tic_id'] for c in passing]}."
            if passed
            else "No candidate satisfied Keplerian orbital stellar density consistency."
        ),
    )


def verify_ccd_centroid_lock(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    params: dict[str, Any] | None = None,
) -> VerifierResult:
    params = params or {}
    max_offset_px = float(params.get("max_offset_px", 0.45))
    min_pix_snr = float(params.get("min_pixel_snr", 5.5))
    min_crowding = float(params.get("min_crowding_metric", 0.80))

    all_cands = _query_all_candidate_metrics(con, data_dir, tic_filter=None)
    snr_cands = [
        c for c in all_cands if c["n_transits"] >= 2 and c["transit_snr"] >= 7.0 and c["depth_ppm"] >= 150.0
    ]
    passing = [
        {
            "tic_id": c["tic_id"],
            "centroid_offset_px": c["centroid_offset_px"],
            "peak_pixel_snr": c["peak_pixel_snr"],
            "crowding_metric": c["crowding_metric"],
        }
        for c in snr_cands
        if c["centroid_offset_px"] <= max_offset_px
        and c["peak_pixel_snr"] >= min_pix_snr
        and c["crowding_metric"] >= min_crowding
    ]
    flagged_spatial = [
        {
            "tic_id": c["tic_id"],
            "centroid_offset_px": c["centroid_offset_px"],
            "peak_pixel_snr": c["peak_pixel_snr"],
            "crowding_metric": c["crowding_metric"],
            "spatial_issue": (
                "off_target_background_binary"
                if c["centroid_offset_px"] > max_offset_px and c["peak_pixel_snr"] >= min_pix_snr
                else "crowded_or_low_snr_ambiguous"
            ),
        }
        for c in snr_cands
        if not (
            c["centroid_offset_px"] <= max_offset_px
            and c["peak_pixel_snr"] >= min_pix_snr
            and c["crowding_metric"] >= min_crowding
        )
    ]

    passed = len(passing) > 0
    return VerifierResult(
        verifier_name="verify_ccd_centroid_lock",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={"candidates": passing, "spatial_flags": flagged_spatial},
        artifacts=[],
        row_count=len(passing),
        summary=(
            f"On-target CCD pixel centroid lock certified for {[c['tic_id'] for c in passing]}."
            if passed
            else f"UNCERTIFIED spatial lock: no candidate passed CCD centroid/crowding criteria ({flagged_spatial})."
        ),
    )


def verify_exoplanet_confirmation(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    params: dict[str, Any] | None = None,
) -> VerifierResult:
    params = params or {}

    all_cands = _query_all_candidate_metrics(con, data_dir, tic_filter=None)
    snr_cands = [
        c for c in all_cands if c["n_transits"] >= 2 and c["transit_snr"] >= 7.0 and c["depth_ppm"] >= 150.0
    ]

    confirmed: list[dict[str, Any]] = []
    rejected_or_ambiguous: list[dict[str, Any]] = []

    for c in snr_cands:
        ok_oe = c["odd_even_sigma"] < 3.0 and c["secondary_snr"] < 4.5
        ok_dens = 0.18 <= c["density_ratio"] <= 5.5
        ok_cent = (
            c["centroid_offset_px"] <= 0.45
            and c["peak_pixel_snr"] >= 5.5
            and c["crowding_metric"] >= 0.80
        )
        if ok_oe and ok_dens and ok_cent:
            confirmed.append(c)
        else:
            reasons: list[str] = []
            if not ok_oe:
                reasons.append("odd_even_or_secondary_eclipse")
            if not ok_dens:
                reasons.append("keplerian_density_violation")
            if not ok_cent:
                if c["centroid_offset_px"] > 0.45 and c["peak_pixel_snr"] >= 5.5:
                    reasons.append("off_target_ccd_centroid_shift")
                else:
                    reasons.append("crowded_or_low_snr_aperture")
            rejected_or_ambiguous.append({"tic_id": c["tic_id"], "failed_checks": reasons})

    artifacts: list[ArtifactRef] = []
    artifact_dir = params.get("artifact_dir")
    if artifact_dir:
        targets_to_render = confirmed[:2] if confirmed else snr_cands[:1]
        for cand in targets_to_render:
            artifacts.append(
                render_exoplanet_diagnostic_card(
                    con=con,
                    data_dir=data_dir,
                    tic_id=cand["tic_id"],
                    out_dir=str(artifact_dir),
                )
            )

    passed = len(confirmed) > 0
    return VerifierResult(
        verifier_name="verify_exoplanet_confirmation",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={
            "confirmed_planets": [c["tic_id"] for c in confirmed],
            "candidates": confirmed,
            "rejected_or_ambiguous": rejected_or_ambiguous,
        },
        artifacts=artifacts,
        row_count=len(confirmed),
        summary=(
            f"CERTIFIED 4-way exoplanet confirmation: {[c['tic_id'] for c in confirmed]}"
            if passed
            else f"UNCERTIFIED: Zero candidates satisfied all 4 checks ({rejected_or_ambiguous})."
        ),
    )
