"""Generate `data/tess_staged/cases_manifest.json` from profiled real NASA TESS stars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGED_DIR = REPO_ROOT / "data" / "tess_staged"


def main() -> None:
    prof_path = STAGED_DIR / "profiled_targets.json"
    prof = json.loads(prof_path.read_text())

    by_tic: dict[str, dict[str, Any]] = {}
    for grp in ("confirmed_planets", "vetted_impostors", "ambiguous_candidates"):
        for rec in prof[grp]:
            by_tic[rec["tic_id"]] = rec

    cases = [
        # --- 3x Real NASA Confirmed Planets (multi_clue_join) ---
        {
            "case_id": "tess_mc_01_toi195_planet",
            "category": "multi_clue_join",
            "scenario_type": "confirmed_exoplanet",
            "target_entity": "TIC-230982885",
            "decoy_entity": None,
            "missing_signal": None,
            "subtlety": "medium",
            "description": (
                "Real NASA TESS Confirmed Planet (TOI-195.01 / KP): 12,690 ppm transit dip (SNR=366.3), "
                "symmetric odd/even eclipses (0.26 sigma), Keplerian density ratio 1.09x, and 0.19 px bullseye CCD lock."
            ),
            "expected_decisions": {
                "confirmed_planet_tic": "TIC-230982885",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            "seed": 501,
            "metadata": {
                "blind_title": "Sector 02 · Single Transit Trigger Alpha",
                "sector": by_tic["TIC-230982885"]["sector"],
            },
        },
        {
            "case_id": "tess_mc_02_toi231_planet",
            "category": "multi_clue_join",
            "scenario_type": "confirmed_exoplanet",
            "target_entity": "TIC-204376737",
            "decoy_entity": None,
            "missing_signal": None,
            "subtlety": "medium",
            "description": (
                "Real NASA TESS Confirmed Planet (TOI-231.01 / KP): 22,042 ppm transit (SNR=205.5), "
                "Keplerian density ratio 1.30x, CROWDSAP=1.00, and 0.24 px on-target CCD lock."
            ),
            "expected_decisions": {
                "confirmed_planet_tic": "TIC-204376737",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            "seed": 502,
            "metadata": {
                "blind_title": "Sector 02 · High-Depth Candidate Beta",
                "sector": by_tic["TIC-204376737"]["sector"],
            },
        },
        {
            "case_id": "tess_mc_03_toi1165_planet",
            "category": "multi_clue_join",
            "scenario_type": "confirmed_exoplanet",
            "target_entity": "TIC-458478250",
            "decoy_entity": None,
            "missing_signal": None,
            "subtlety": "low",
            "description": (
                "Real NASA TESS Kepler-Field Planet (TOI-1165.01 / KP): SNR=474.9, odd/even 0.24 sigma, "
                "Keplerian density 1.92x, and 0.28 px CCD pixel lock."
            ),
            "expected_decisions": {
                "confirmed_planet_tic": "TIC-458478250",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            "seed": 503,
            "metadata": {
                "blind_title": "Sector 14 · Northern Field Trigger Gamma",
                "sector": by_tic["TIC-458478250"]["sector"],
            },
        },
        # --- 3x Real NASA Cosmic Impostor Battles (look_alike_decoy) ---
        {
            "case_id": "tess_decoy_01_off_target_ccd_shift",
            "category": "look_alike_decoy",
            "scenario_type": "centroid_shift_impostor",
            "target_entity": "TIC-27848472",
            "decoy_entity": "TIC-219821826",
            "missing_signal": None,
            "subtlety": "high",
            "description": (
                "Deeper-transit NASA False Positive decoy TIC-219821826 (TOI-2258.01 / FP, 13,618 ppm) shifts "
                "0.54 px off-target on the CCD and violates Keplerian density (17x), while real planet "
                "TIC-27848472 (TOI-2131.01 / KP, 11,466 ppm) locks at 0.07 px on-target."
            ),
            "expected_decisions": {
                "confirmed_planet_tic": "TIC-27848472",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            "seed": 601,
            "metadata": {
                "blind_title": "Sector 26 · Dual-Candidate Spatial Contention",
                "sector": by_tic["TIC-27848472"]["sector"],
            },
        },
        {
            "case_id": "tess_decoy_02_odd_even_binary_trap",
            "category": "look_alike_decoy",
            "scenario_type": "odd_even_impostor",
            "target_entity": "TIC-138168780",
            "decoy_entity": "TIC-260271203",
            "missing_signal": None,
            "subtlety": "high",
            "description": (
                "3-of-4 look-alike NASA False Positive TIC-260271203 (TOI-207.01 / FP) has a massive 37,487 ppm dip "
                "(3x deeper than planet!) and passes CCD centroid (0.10 px) & Keplerian density (1.42x), failing ONLY "
                "odd/even eclipse symmetry (4.28 sigma); real planet TIC-138168780 (TOI-1651.01 / KP) passes all 4."
            ),
            "expected_decisions": {
                "confirmed_planet_tic": "TIC-138168780",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            "seed": 602,
            "metadata": {
                "blind_title": "Sector 14 · Giant-Depth Eclipse Contention",
                "sector": by_tic["TIC-138168780"]["sector"],
            },
        },
        {
            "case_id": "tess_decoy_03_kepler_density_impostor",
            "category": "look_alike_decoy",
            "scenario_type": "kepler_density_impostor",
            "target_entity": "TIC-288735205",
            "decoy_entity": "TIC-183979262",
            "missing_signal": None,
            "subtlety": "high",
            "description": (
                "Screaming 527.5-sigma SNR NASA False Positive TIC-183979262 (TOI-183.01 / FP) passes CCD centroid "
                "but violates Kepler's 3rd Law stellar density by 16.1x with a 35.8-sigma secondary occultation; "
                "real planet TIC-288735205 (TOI-1259.01 / CP, SNR=229.3) passes all 4 checks."
            ),
            "expected_decisions": {
                "confirmed_planet_tic": "TIC-288735205",
                "recommended_followup": "trigger_flagship_slew",
                "sufficiently_verified": "yes",
            },
            "seed": 603,
            "metadata": {
                "blind_title": "Sector 14 · Extreme-SNR Multi-Star Alert",
                "sector": by_tic["TIC-288735205"]["sector"],
            },
        },
        # --- 3x Real NASA Unconfirmed / Crowded Candidates (under_determined) ---
        {
            "case_id": "tess_ud_01_crowded_aperture_toi1832",
            "category": "under_determined",
            "scenario_type": "crowded_field_candidate",
            "target_entity": "TIC-307956397",
            "decoy_entity": None,
            "missing_signal": "ccd_centroid_lock",
            "subtlety": "medium",
            "description": (
                "Real NASA Unconfirmed Candidate TIC-307956397 (TOI-1832.01 / PC): passes transit SNR (22.5), "
                "odd/even symmetry (1.65 sigma), and Keplerian density (1.54x), but CROWDSAP=0.57 blends neighbor "
                "starlight so spatial lock is UNCERTIFIED -> must request spectroscopy without false certainty."
            ),
            "expected_decisions": {
                "confirmed_planet_tic": "none_verified",
                "recommended_followup": "request_spectroscopy",
                "sufficiently_verified": "no",
            },
            "seed": 701,
            "metadata": {
                "blind_title": "Sector 26 · Crowded Cluster Alert Delta",
                "sector": by_tic["TIC-307956397"]["sector"],
            },
        },
        {
            "case_id": "tess_ud_02_ambiguous_candidate_toi182",
            "category": "under_determined",
            "scenario_type": "ambiguous_pc",
            "target_entity": "TIC-76989773",
            "decoy_entity": None,
            "missing_signal": "ccd_centroid_lock",
            "subtlety": "high",
            "description": (
                "Real NASA Ambiguous Planetary Candidate TIC-76989773 (TOI-182.01 / APC): strong 188.7-sigma "
                "periodic dip passing odd/even & Keplerian density, but CROWDSAP=0.77 fails aperture isolation "
                "threshold -> must halt honestly as UNCERTIFIED and request follow-up spectroscopy."
            ),
            "expected_decisions": {
                "confirmed_planet_tic": "none_verified",
                "recommended_followup": "request_spectroscopy",
                "sufficiently_verified": "no",
            },
            "seed": 702,
            "metadata": {
                "blind_title": "Sector 01 · Bright Blended Candidate Epsilon",
                "sector": by_tic["TIC-76989773"]["sector"],
            },
        },
        {
            "case_id": "tess_ud_03_faint_stamp_toi1825",
            "category": "under_determined",
            "scenario_type": "faint_stamp_candidate",
            "target_entity": "TIC-229750058",
            "decoy_entity": None,
            "missing_signal": "ccd_centroid_lock",
            "subtlety": "medium",
            "description": (
                "Real NASA Planetary Candidate TIC-229750058 (TOI-1825.01 / PC): passes 1D transit SNR (38.0), "
                "odd/even (0.12 sigma), and Keplerian density (3.92x), but lacks per-pixel difference SNR to "
                "certify sub-pixel CCD centroid lock -> must remain UNCERTIFIED."
            ),
            "expected_decisions": {
                "confirmed_planet_tic": "none_verified",
                "recommended_followup": "request_spectroscopy",
                "sufficiently_verified": "no",
            },
            "seed": 703,
            "metadata": {
                "blind_title": "Sector 14 · Faint Pixel-Stamp Candidate Zeta",
                "sector": by_tic["TIC-229750058"]["sector"],
            },
        },
    ]

    candidate_tics_ordered: list[str] = []
    nasa_truth: dict[str, dict[str, Any]] = {}
    for c in cases:
        for k in ("target_entity", "decoy_entity"):
            tic = c.get(k)
            if tic and tic not in candidate_tics_ordered:
                candidate_tics_ordered.append(tic)
                r = by_tic[tic]
                nasa_truth[tic] = {
                    "tic_id": tic,
                    "toi": f"TOI-{r['toi']}",
                    "official_nasa_disposition": r["tfopwg_disp"],
                    "disposition_meaning": {
                        "CP": "Confirmed Exoplanet (NASA TFOP Verified)",
                        "KP": "Known Confirmed Exoplanet",
                        "FP": "Astrophysical False Positive (Cosmic Impostor)",
                        "PC": "Unconfirmed Planetary Candidate (Follow-Up Required)",
                        "APC": "Ambiguous Planetary Candidate (Blended / Uncertain)",
                    }.get(r["tfopwg_disp"], r["tfopwg_disp"]),
                    "sector": r["sector"],
                    "period_days": r["pl_orbper"],
                    "catalog_depth_ppm": r["pl_trandep"],
                    "planet_radius_rearth": r.get("pl_rade"),
                    "tmag": r["st_tmag"],
                    "stellar_radius_rsun": r["st_rad"],
                    "teff_k": r["st_teff"],
                    "ra_deg": r["ra"],
                    "dec_deg": r["dec"],
                    "measured_physics": r["physics"],
                }

    out_path = STAGED_DIR / "cases_manifest.json"
    out_path.write_text(
        json.dumps(
            {
                "domain": "nasa_tess_exoplanet_triage",
                "background_stars": prof["background_stars"],
                "background_cadences": prof["background_cadences"],
                "candidate_tics": candidate_tics_ordered,
                "cases": cases,
                "nasa_archive_truth": nasa_truth,
            },
            indent=2,
        )
    )
    print(
        f"[MANIFEST] Wrote {len(cases)} real NASA TESS benchmark cases "
        f"({len(candidate_tics_ordered)} unique TICs) -> {out_path}"
    )


if __name__ == "__main__":
    main()
