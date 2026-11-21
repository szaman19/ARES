"""Safe real-world multimodal datasets adapted to `origin/main` ARES `ExecutionBackend`.

Connects staged `.data_cache/<dataset_id>` slices (ZTF `btsbot` supernovae,
Caltech Camera Traps, Pyronear wildfire lookout towers, and UrbanSound8K audio)
to `DuckDBBackend`, `ares.solve()`, `NoteStore`, `validate_and_merge_patch()`,
`rebuild_board_from_notes()`, `GeminiModelProvider`, and `DecisionProbeClient`
with zero edits to `ares/`, `contracts.py`, `fusion/`, or `models/`.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import re
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import httpx

from adapters.duckdb_backend import DuckDBBackend, QueryExecutionResult
from adapters.sql_guard import SQLGuardConfig, SQLValidationError
from ares.engine import TaskGraph, TaskSpec, solve
from ares.store import NoteStore
from contracts import (
    ArtifactKind,
    ArtifactRef,
    DecisionQuestion,
    InMemoryVerifierRegistry,
    Note,
    VerifierRegistry,
    VerifierResult,
)

ALLOWED_HTTPS_HOSTS: frozenset[str] = frozenset(
    {
        "huggingface.co",
        "datasets-server.huggingface.co",
        "cas-bridge.xethub.hf.co",
        "cdn-lfs.hf.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs-eu-1.hf.co",
        "storage.googleapis.com",
        "lila.science",
        "lilawildlife.blob.core.windows.net",
    }
)

ALLOWED_SAFE_EXTENSIONS: frozenset[str] = frozenset(
    {".parquet", ".csv", ".json", ".png", ".jpg", ".jpeg", ".wav", ".mp3", ".mp4"}
)

FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".ckpt",
        ".bin",
        ".sh",
        ".bash",
        ".zsh",
        ".py",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
    }
)

DEFAULT_MAX_MEDIA_BYTES: int = 25 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_UNCOMPRESSED_BYTES: int = 220 * 1024 * 1024

CANONICAL_MULTIMODAL_FIELDS: tuple[str, ...] = (
    "primary_tabular_trigger",
    "contextual_telemetry",
    "visual_media_evidence",
)


class SafetyViolationError(ValueError):
    """Raised when a URL, path, archive member, or binary header violates safety rules."""


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def verify_safe_https_url(url: str) -> str:
    """Validate that `url` uses HTTPS and points to an explicitly allowlisted host."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise SafetyViolationError(f"Only https:// URLs are permitted, got: {url!r}")
    host = (parsed.hostname or "").lower()
    is_allowed = host in ALLOWED_HTTPS_HOSTS or host.endswith(".hf.co")
    if not is_allowed:
        raise SafetyViolationError(
            f"Untrusted host {host!r} in URL {url!r}. Allowed: {sorted(ALLOWED_HTTPS_HOSTS)}"
        )
    lower_path = parsed.path.lower()
    for bad_ext in FORBIDDEN_EXTENSIONS:
        if lower_path.endswith(bad_ext):
            raise SafetyViolationError(
                f"Forbidden unsafe file extension {bad_ext!r} in URL {url!r}"
            )
    return url


def verify_safe_output_path(path: Path, root_dir: Path) -> Path:
    """Ensure `path` resolves strictly inside `root_dir` and has a safe extension."""
    resolved_root = root_dir.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise SafetyViolationError(
            f"Refusing to access path outside cache root {resolved_root}: {resolved_path}"
        ) from exc

    suffix = resolved_path.suffix.lower()
    if suffix in FORBIDDEN_EXTENSIONS:
        raise SafetyViolationError(f"Forbidden unsafe file extension: {suffix!r}")
    if suffix and suffix not in ALLOWED_SAFE_EXTENSIONS:
        raise SafetyViolationError(
            f"Extension {suffix!r} is not in safe allowlist {sorted(ALLOWED_SAFE_EXTENSIONS)}"
        )
    return resolved_path


def verify_magic_bytes(
    data: bytes,
    expected_ext: str,
    *,
    max_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
) -> None:
    """Verify magic-byte signatures and size bounds on downloaded/local media."""
    if not data:
        raise SafetyViolationError("Empty payload rejected.")
    if len(data) > max_bytes:
        raise SafetyViolationError(
            f"Payload size {len(data)} bytes exceeds safety limit of {max_bytes} bytes."
        )
    ext = expected_ext.lower()
    if ext == ".png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise SafetyViolationError("Invalid PNG magic header.")
    elif ext in (".jpg", ".jpeg"):
        if not data.startswith(b"\xff\xd8\xff"):
            raise SafetyViolationError("Invalid JPEG SOI magic header.")
    elif ext == ".wav":
        if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise SafetyViolationError("Invalid RIFF/WAVE audio header.")
    elif ext == ".parquet":
        if len(data) < 12 or data[:4] != b"PAR1" or data[-4:] != b"PAR1":
            raise SafetyViolationError("Invalid Apache Parquet PAR1 magic bytes.")
    elif ext == ".json":
        stripped = data.lstrip()
        if not stripped or stripped[:1] not in (b"{", b"["):
            raise SafetyViolationError("Invalid JSON payload start token.")
    else:
        raise SafetyViolationError(f"Unsupported verification extension: {ext!r}")


def safe_https_get_bytes(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
    timeout_s: float = 30.0,
) -> bytes:
    """Download bytes over HTTPS enforcing redirect host checks and hard byte ceilings."""
    verify_safe_https_url(url)
    chunks: list[bytes] = []
    total = 0
    with httpx.Client(timeout=timeout_s, follow_redirects=False) as client:
        current_url = url
        for _hop in range(6):
            verify_safe_https_url(current_url)
            with client.stream("GET", current_url) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location")
                    if not loc:
                        raise SafetyViolationError(
                            f"Redirect missing Location header from {current_url}"
                        )
                    current_url = urllib.parse.urljoin(current_url, loc)
                    continue
                resp.raise_for_status()
                cl = resp.headers.get("content-length")
                if cl is not None and int(cl) > max_bytes:
                    raise SafetyViolationError(
                        f"Remote Content-Length {cl} exceeds cap {max_bytes} for {current_url}"
                    )
                for chunk in resp.iter_bytes(chunk_size=65536):
                    total += len(chunk)
                    if total > max_bytes:
                        raise SafetyViolationError(
                            f"Stream exceeded cap {max_bytes} bytes while fetching {current_url}"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
    raise SafetyViolationError(f"Too many HTTP redirects while fetching {url}")


def safe_extract_single_json_from_zip(
    zip_bytes: bytes,
    *,
    max_uncompressed_bytes: int = DEFAULT_MAX_ARCHIVE_UNCOMPRESSED_BYTES,
) -> bytes:
    """Safely extract a single `.json` document from an in-memory ZIP archive."""
    if not zip_bytes.startswith(b"PK\x03\x04"):
        raise SafetyViolationError("Invalid ZIP header.")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if not infos:
            raise SafetyViolationError("ZIP archive contains no files.")
        json_infos: list[zipfile.ZipInfo] = []
        for info in infos:
            norm = posixpath.normpath(info.filename)
            if norm.startswith("/") or norm.startswith("..") or "/../" in f"/{norm}/":
                raise SafetyViolationError(f"Zip-slip path traversal blocked: {info.filename!r}")
            ext = Path(norm).suffix.lower()
            if ext != ".json":
                raise SafetyViolationError(
                    f"Non-JSON member {info.filename!r} rejected inside metadata ZIP."
                )
            if info.file_size > max_uncompressed_bytes:
                raise SafetyViolationError(
                    f"Uncompressed JSON size {info.file_size} exceeds cap {max_uncompressed_bytes}."
                )
            json_infos.append(info)
        target = json_infos[0]
        out = io.BytesIO()
        total = 0
        with zf.open(target, "r") as member:
            while True:
                buf = member.read(65536)
                if not buf:
                    break
                total += len(buf)
                if total > max_uncompressed_bytes:
                    raise SafetyViolationError("Decompression exceeded uncompressed byte ceiling.")
                out.write(buf)
        raw = out.getvalue()
        verify_magic_bytes(raw, ".json", max_bytes=max_uncompressed_bytes)
        return raw


@dataclass(frozen=True)
class PresetMagnet:
    """Clickable SQL magnet preset for the interactive demo."""

    label: str
    description: str
    sql: str


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata, schemas, rubrics, and SQL presets for one multimodal dataset."""

    dataset_id: str
    short_label: str
    emoji: str
    title: str
    domain: str
    source_url: str
    primary_media_kind: str
    positive_option: str
    positive_label_description: str
    decoy_description: str
    schema_summary: str
    hypothesis_families: tuple[str, ...]
    decision_questions: tuple[DecisionQuestion, ...]
    preset_magnets: tuple[PresetMagnet, ...]
    preview_columns: tuple[str, ...]


DATASET_CATALOG: dict[str, DatasetSpec] = {
    "caltech_camera_traps": DatasetSpec(
        dataset_id="caltech_camera_traps",
        short_label="Wildlife Camera Traps",
        emoji="🐾",
        title="Caltech Camera Traps + MegaDetector v5 (LILA BC)",
        domain="Wildlife Ecology & Motion Sensors",
        source_url="https://lila.science/datasets/caltech-camera-traps/",
        primary_media_kind="Daytime RGB & Night IR Camera-Trap Photos (.jpg)",
        positive_option="real_wildlife",
        positive_label_description="Confirmed wild mammal/bird in frame (`bobcat`, `coyote`, `deer`, `fox`, `raccoon`)",
        decoy_description=(
            "Natural false triggers (`empty`): wind-whipped desert brush, sun glare, and "
            "rock shadows that fool automated motion & MegaDetector scores (`conf >= 0.80`)"
        ),
        schema_summary=(
            "Table `observations` (243,100 Southwest US camera-trap captures):\n"
            "  - record_id (VARCHAR), location_id (VARCHAR), captured_at (VARCHAR), hour_local (INT)\n"
            "  - seq_id (VARCHAR), seq_num_frames (INT), frame_num (INT), width (INT), height (INT)\n"
            "  - max_detection_conf (DOUBLE): Automated MegaDetector v5a confidence in [0, 1]\n"
            "  - n_detections (INT), max_bbox_area_frac (DOUBLE)\n"
            "  - media_url (VARCHAR): Direct HTTPS URL for on-demand `.jpg` inspection\n"
            "  - media_relpath (VARCHAR): Pre-cached local `.jpg` relative path"
        ),
        hypothesis_families=(
            "high_conf_predator_sequence",
            "thermal_wind_brush_decoy",
            "nocturnal_wildlife_corridor",
        ),
        decision_questions=(
            DecisionQuestion(
                question_id="camera_trigger_cause",
                question="Using BOTH the tabular detector metadata and the attached camera-trap photo (if present), what caused this trigger?",
                options=["real_wildlife", "empty_false_trigger", "insufficient_evidence"],
                question_type="multiple_choice",
                instructions=(
                    "Automated MegaDetector confidence (`max_detection_conf`) frequently false-alarms "
                    "above 0.80 on wind-blown desert shrubs, glare, and shadows. If the camera-trap "
                    "image is attached, inspect the scene carefully for an actual animal body/eyes/fur "
                    "versus empty brush/rocks. If visual_media_evidence is 'unknown' (no image attached), "
                    "avoid overconfident wildlife claims because tabular detector numbers alone are ambiguous."
                ),
                option_criteria={
                    "real_wildlife": "Visible wild animal (coyote, bobcat, deer, fox, raccoon, etc.) confirmed in frame.",
                    "empty_false_trigger": "Frame shows only vegetation, rocks, shadows, or camera glare with no animal.",
                    "insufficient_evidence": "Visual photo is missing/'unknown' so tabular detector trigger cannot be disambiguated from a wind/shadow decoy.",
                },
            ),
            DecisionQuestion(
                question_id="sufficiently_verified",
                question="Is a real wild animal visually verified with certified photographic evidence?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ),
        preset_magnets=(
            PresetMagnet(
                label="High-Confidence Detector Hits (Cached Photos)",
                description="Filter 243,100 triggers to `max_detection_conf >= 0.80` with instant cached `.jpg` frames (mix of real animals & empty brush decoys!).",
                sql=(
                    "SELECT record_id, location_id, captured_at, hour_local, seq_num_frames, "
                    "ROUND(max_detection_conf, 3) AS max_detection_conf, n_detections, "
                    "ROUND(max_bbox_area_frac, 3) AS max_bbox_area_frac, media_relpath "
                    "FROM observations "
                    "WHERE max_detection_conf >= 0.80 AND media_relpath != '' "
                    "ORDER BY max_detection_conf DESC LIMIT 16"
                ),
            ),
            PresetMagnet(
                label="Live On-Demand Search Across All 243,100 Rows",
                description="Query the entire 243k-row catalog (even uncached rows! Clicking any row safely fetches its `.jpg` over HTTPS on demand).",
                sql=(
                    "SELECT record_id, location_id, captured_at, hour_local, seq_num_frames, "
                    "ROUND(max_detection_conf, 3) AS max_detection_conf, n_detections, "
                    "ROUND(max_bbox_area_frac, 3) AS max_bbox_area_frac, media_relpath "
                    "FROM observations "
                    "WHERE max_detection_conf BETWEEN 0.82 AND 0.96 AND seq_num_frames >= 3 "
                    "ORDER BY seq_num_frames DESC, max_detection_conf DESC LIMIT 12"
                ),
            ),
        ),
        preview_columns=(
            "record_id",
            "location_id",
            "captured_at",
            "seq_num_frames",
            "max_detection_conf",
            "max_bbox_area_frac",
        ),
    ),
    "btsbot": DatasetSpec(
        dataset_id="btsbot",
        short_label="ZTF Supernova Alerts",
        emoji="🔭",
        title="ZTF BTSbot Supernova Transient Alerts (MultimodalUniverse)",
        domain="Astronomy / Time-Domain Sky Survey",
        source_url="https://huggingface.co/datasets/MultimodalUniverse/btsbot",
        primary_media_kind="3-Panel Telescope Cutouts: Science | Reference | Difference (.png)",
        positive_option="real_supernova",
        positive_label_description="Spectroscopically confirmed extragalactic supernova (`is_SN = True`)",
        decoy_description=(
            "High-`drb` alert decoys: recurring variable stars, AGN nuclear flares (`ndethist > 50`), "
            "cosmic rays, and dipole CCD subtraction ghosts"
        ),
        schema_summary=(
            "Table `observations` (1,200 Zwicky Transient Facility sky alerts):\n"
            "  - record_id (VARCHAR), object_id (VARCHAR), jd (DOUBLE)\n"
            "  - magpsf, sigmapsf, diffmaglim, dsnrms, ssnrms (DOUBLE)\n"
            "  - drb, new_drb (DOUBLE): Deep real-bogus score in [0, 1] (>0.99 on BOTH real SNe & decoys!)\n"
            "  - fwhm, chipsf, nbad, nneg, mindtoedge (PSF & CCD subtraction diagnostics)\n"
            "  - sgscore1 (0=extended galaxy, 1=point star), distpsnr1 (host offset arcsec)\n"
            "  - ndethist, age, days_since_peak, acai_h, acai_v, acai_n, acai_b, is_rise"
        ),
        hypothesis_families=(
            "extragalactic_rising_supernova",
            "nuclear_agn_or_variable_star_decoy",
            "ccd_dipole_subtraction_ghost",
        ),
        decision_questions=(
            DecisionQuestion(
                question_id="transient_classification",
                question="Based on the alert telemetry and the 3-panel Science/Reference/Difference telescope stamps, what is this sky transient?",
                options=["real_supernova", "variable_or_artifact_decoy", "insufficient_evidence"],
                question_type="multiple_choice",
                instructions=(
                    "A real extragalactic supernova typically shows a clean circular point source at "
                    "the crosshair in Science & Difference, low detection history (`ndethist` small / young `age`), "
                    "and often an extended host galaxy (`sgscore1 < 0.5`). Recurring variable stars / AGN flares "
                    "often have long detection histories (`ndethist >> 30`), stellar `sgscore1`, or dipole/streaked "
                    "Difference subtraction artifacts. If `visual_media_evidence` is 'unknown', choose "
                    "`insufficient_evidence` whenever tabular signals conflict or subtraction quality is unverified."
                ),
                option_criteria={
                    "real_supernova": "Clean positive point-source residual in Difference cutout + young extragalactic transient profile.",
                    "variable_or_artifact_decoy": "Long recurring history (variable/AGN), stellar counterpart, or bad Difference subtraction residual.",
                    "insufficient_evidence": "Sky cutout triplet is missing/'unknown' so image-subtraction artifacts cannot be ruled out.",
                },
            ),
            DecisionQuestion(
                question_id="sufficiently_verified",
                question="Should scarce spectroscopic telescope time be triggered based on verified visual + photometric evidence?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ),
        preset_magnets=(
            PresetMagnet(
                label="High-DRB Bright Alerts (`drb > 0.99`, Mix of SNe & Decoys)",
                description="Alerts where ML real-bogus `drb > 0.99` says 'real!' — contains both newborn supernovae and long-history variable/AGN decoys.",
                sql=(
                    "SELECT record_id, object_id, ROUND(magpsf, 2) AS magpsf, "
                    "ROUND(drb, 4) AS drb, ndethist, ROUND(sgscore1, 3) AS sgscore1, "
                    "ROUND(distpsnr1, 2) AS distpsnr1, nbad, media_relpath "
                    "FROM observations "
                    "WHERE drb > 0.99 AND media_relpath != '' "
                    "ORDER BY magpsf ASC LIMIT 16"
                ),
            ),
            PresetMagnet(
                label="Young Extragalactic Candidates (`sgscore1 < 0.35`, `ndethist <= 10`)",
                description="Multi-clue SQL magnet targeting galaxy-hosted (`sgscore1 < 0.35`), newly erupted (`ndethist <= 10`) transients.",
                sql=(
                    "SELECT record_id, object_id, ROUND(magpsf, 2) AS magpsf, "
                    "ROUND(drb, 4) AS drb, ndethist, ROUND(sgscore1, 3) AS sgscore1, "
                    "ROUND(distpsnr1, 2) AS distpsnr1, nbad, media_relpath "
                    "FROM observations "
                    "WHERE sgscore1 < 0.35 AND ndethist <= 10 AND media_relpath != '' "
                    "ORDER BY drb DESC, magpsf ASC LIMIT 12"
                ),
            ),
        ),
        preview_columns=(
            "record_id",
            "object_id",
            "magpsf",
            "drb",
            "ndethist",
            "sgscore1",
        ),
    ),
    "pyro_sdis": DatasetSpec(
        dataset_id="pyro_sdis",
        short_label="Wildfire Lookout Towers",
        emoji="🔥",
        title="Pyronear Early Wildfire Lookout Network (pyro-sdis)",
        domain="Environmental / Wildfire Early Warning",
        source_url="https://huggingface.co/datasets/pyronear/pyro-sdis",
        primary_media_kind="Mountaintop Fire-Lookout Tower Photos (.jpg)",
        positive_option="wildfire_smoke_plume",
        positive_label_description="Genuine rising wildfire smoke plume (`has_smoke = True`)",
        decoy_description=(
            "Hard-negative lookout false alarms: low valley fog banks, cloud decks, "
            "dust, and solar lens glare spiking optical haze metrics"
        ),
        schema_summary=(
            "Table `observations` (1,129 mountaintop fire-lookout tower alert frames):\n"
            "  - record_id, partner, camera, azimuth_deg (INT), captured_at, hour_obs (INT)\n"
            "  - width, height, jpg_bytes (INT)\n"
            "  - brightness_mean, brightness_std, horizon_brightness, sky_ground_contrast (DOUBLE)\n"
            "  - gray_haze_fraction (DOUBLE): Low-saturation bright haze fraction (high on BOTH smoke & fog!)\n"
            "  - warm_glare_fraction (DOUBLE): Solar glare indicator"
        ),
        hypothesis_families=(
            "localized_rising_smoke_plume",
            "valley_fog_or_cloud_deck_decoy",
            "solar_glare_optical_false_alarm",
        ),
        decision_questions=(
            DecisionQuestion(
                question_id="lookout_alert_assessment",
                question="Based on the tower optical telemetry and mountaintop lookout photograph, what triggered this fire-watch alert?",
                options=["wildfire_smoke_plume", "fog_cloud_or_glare_decoy", "insufficient_evidence"],
                question_type="multiple_choice",
                instructions=(
                    "Inspect the lookout tower photograph carefully for a localized rising smoke plume "
                    "originating from terrain/forest canopy, versus uniform cloud decks, valley fog, "
                    "clear sky, or sun glare. Tabular `gray_haze_fraction` and `horizon_brightness` "
                    "cannot distinguish smoke from clouds/fog without the visual frame; if "
                    "`visual_media_evidence` is 'unknown', select `insufficient_evidence`."
                ),
                option_criteria={
                    "wildfire_smoke_plume": "Localized smoke plume rising from ground/hills visible in the tower photograph.",
                    "fog_cloud_or_glare_decoy": "Photograph shows only clouds, mist/fog, glare, or clear landscape with no wildfire plume.",
                    "insufficient_evidence": "Lookout photograph is 'unknown'/missing so optical haze numbers cannot separate smoke from clouds.",
                },
            ),
            DecisionQuestion(
                question_id="sufficiently_verified",
                question="Is a real smoke plume visually confirmed to dispatch fire-watch verification?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ),
        preset_magnets=(
            PresetMagnet(
                label="High Horizon-Haze Tower Triggers (Smoke vs. Cloud/Fog Decoys)",
                description="Tower frames with elevated `gray_haze_fraction >= 0.12` — optical numbers look identical on real smoke and cloud/fog false alarms!",
                sql=(
                    "SELECT record_id, partner, camera, captured_at, "
                    "ROUND(gray_haze_fraction, 3) AS gray_haze_fraction, "
                    "ROUND(horizon_brightness, 3) AS horizon_brightness, "
                    "ROUND(sky_ground_contrast, 3) AS sky_ground_contrast, media_relpath "
                    "FROM observations "
                    "WHERE gray_haze_fraction >= 0.12 AND media_relpath != '' "
                    "ORDER BY gray_haze_fraction DESC LIMIT 16"
                ),
            ),
            PresetMagnet(
                label="Afternoon Lookout Alerts Across Tower Partners",
                description="Browse cached lookout captures across `biobio`, `engie`, and `sdis` fire towers.",
                sql=(
                    "SELECT record_id, partner, camera, captured_at, "
                    "ROUND(gray_haze_fraction, 3) AS gray_haze_fraction, "
                    "ROUND(warm_glare_fraction, 3) AS warm_glare_fraction, "
                    "ROUND(sky_ground_contrast, 3) AS sky_ground_contrast, media_relpath "
                    "FROM observations "
                    "WHERE media_relpath != '' "
                    "ORDER BY captured_at DESC LIMIT 16"
                ),
            ),
        ),
        preview_columns=(
            "record_id",
            "partner",
            "camera",
            "gray_haze_fraction",
            "horizon_brightness",
            "sky_ground_contrast",
        ),
    ),
    "urbansound8k": DatasetSpec(
        dataset_id="urbansound8k",
        short_label="Urban Acoustic Sensors",
        emoji="🔊",
        title="UrbanSound8K Acoustic Sensor Network (.wav + Spectrogram)",
        domain="Urban Audio Sensors + Acoustic Telemetry",
        source_url="https://huggingface.co/datasets/danavery/urbansound8K",
        primary_media_kind="Playable Field Audio (.wav) + Waveform/Spectrogram Card (.png)",
        positive_option="emergency_gunshot_or_siren",
        positive_label_description="High-priority emergency acoustic event (`gun_shot` or `siren`)",
        decoy_description=(
            "Impulsive urban noise decoys (`jackhammer`, `car_horn`, `dog_bark`, `drilling`) "
            "whose high peak-to-RMS `crest_factor_db` mimics gunfire or sirens in tabular sensor triggers"
        ),
        schema_summary=(
            "Table `observations` (450 urban acoustic microphone triggers):\n"
            "  - record_id, slice_file_name, freesound_id (BIGINT), duration_s (DOUBLE), salience (1=fg, 2=bg)\n"
            "  - sample_rate_hz, channels (INT), rms_db, peak_amp, crest_factor_db (DOUBLE)\n"
            "  - zero_crossing_rate, spectral_centroid_hz, high_freq_energy_ratio (DOUBLE)\n"
            "  - wav_relpath (VARCHAR): Verified `.wav` clip; media_relpath (VARCHAR): Waveform+Spectrogram `.png`"
        ),
        hypothesis_families=(
            "impulsive_ballistic_transient",
            "sweeping_harmonic_siren",
            "repetitive_construction_or_bark_decoy",
        ),
        decision_questions=(
            DecisionQuestion(
                question_id="acoustic_event_type",
                question="Using the acoustic sensor metrics and the attached waveform + spectrogram card, what type of urban sound is this?",
                options=["emergency_gunshot_or_siren", "urban_noise_decoy", "insufficient_evidence"],
                question_type="multiple_choice",
                instructions=(
                    "In the 2-panel waveform/spectrogram card: a `gun_shot` shows an abrupt broadband "
                    "vertical impulse spike with rapid decay (often short duration); a `siren` shows "
                    "smooth continuous frequency-modulated harmonic bands; `jackhammer`/`drilling` shows "
                    "rapidly repeating mechanical pulses across the 4s window; `dog_bark` shows short "
                    "mid-frequency vocal formants; `car_horn` shows steady flat harmonic tones. "
                    "If `visual_media_evidence` (spectrogram card) is 'unknown', choose `insufficient_evidence` "
                    "because scalar `crest_factor_db` and `rms_db` overlap heavily with urban noise decoys."
                ),
                option_criteria={
                    "emergency_gunshot_or_siren": "Single ballistic impulse decay or continuous FM sweeping siren harmonics confirmed in spectrogram.",
                    "urban_noise_decoy": "Repetitive construction impacts, flat car-horn harmonics, vocal dog barks, or ambient street clatter.",
                    "insufficient_evidence": "Waveform/spectrogram card is 'unknown'/missing so scalar impulse metrics cannot rule out urban noise decoys.",
                },
            ),
            DecisionQuestion(
                question_id="sufficiently_verified",
                question="Is an emergency acoustic signature verified strongly enough to escalate an alert?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ),
        preset_magnets=(
            PresetMagnet(
                label="High Crest-Factor Impulsive Spikes (`crest_factor_db >= 14.5 dB`)",
                description="Triggers with sharp peak-to-RMS spikes (`>= 14.5 dB`) — mixes real `gun_shot`/`siren` clips with `dog_bark` & `jackhammer` decoys!",
                sql=(
                    "SELECT record_id, duration_s, salience, ROUND(rms_db, 1) AS rms_db, "
                    "ROUND(crest_factor_db, 2) AS crest_factor_db, "
                    "ROUND(spectral_centroid_hz, 0) AS spectral_centroid_hz, "
                    "wav_relpath, media_relpath "
                    "FROM observations "
                    "WHERE crest_factor_db >= 14.5 AND media_relpath != '' "
                    "ORDER BY crest_factor_db DESC LIMIT 16"
                ),
            ),
            PresetMagnet(
                label="Loud Foreground Acoustic Events (`salience = 1`, `rms_db > -18`)",
                description="High-energy foreground triggers with playable `.wav` clips and spectrogram cards.",
                sql=(
                    "SELECT record_id, duration_s, salience, ROUND(rms_db, 1) AS rms_db, "
                    "ROUND(crest_factor_db, 2) AS crest_factor_db, "
                    "ROUND(spectral_centroid_hz, 0) AS spectral_centroid_hz, "
                    "wav_relpath, media_relpath "
                    "FROM observations "
                    "WHERE salience = 1 AND rms_db > -18.0 AND media_relpath != '' "
                    "ORDER BY rms_db DESC LIMIT 16"
                ),
            ),
        ),
        preview_columns=(
            "record_id",
            "duration_s",
            "rms_db",
            "crest_factor_db",
            "spectral_centroid_hz",
            "salience",
        ),
    ),
}


def extract_multimodal_note_patches(note: Note) -> dict[str, Any] | None:
    """Deterministic `Note -> board field` extractor for `fusion/patch.py` (`0.0` drift)."""
    vname = note.verifier_name or ""
    ev = note.evidence or {}
    if vname == "verify_primary_sensor_trigger" and "trigger_summary" in ev:
        return {"primary_tabular_trigger": ev["trigger_summary"]}
    if vname == "verify_contextual_telemetry" and "context_summary" in ev:
        return {"contextual_telemetry": ev["context_summary"]}
    if vname == "verify_candidate_media" and "media_summary" in ev:
        return {"visual_media_evidence": ev["media_summary"]}
    return None


class MultimodalSearchDataset:
    """Adapter connecting staged `.data_cache/<dataset_id>` to `DuckDBBackend` & `ares.solve()`."""

    def __init__(self, spec: DatasetSpec, dataset_dir: Path) -> None:
        self.spec = spec
        self.name = spec.dataset_id
        self.dataset_dir = dataset_dir.resolve()
        self.observations_dir = self.dataset_dir / "observations"
        self.media_dir = self.dataset_dir / "media"
        self.audio_dir = self.dataset_dir / "audio"
        self.ground_truth_path = self.dataset_dir / "ground_truth.json"
        self.manifest_path = self.dataset_dir / "manifest.json"

    @classmethod
    def open(
        cls,
        dataset_id: str,
        *,
        cache_root: str | Path = ".data_cache",
    ) -> "MultimodalSearchDataset":
        if dataset_id not in DATASET_CATALOG:
            raise KeyError(
                f"Unknown dataset_id {dataset_id!r}. Choose from: {sorted(DATASET_CATALOG)}"
            )
        spec = DATASET_CATALOG[dataset_id]
        ds_dir = Path(cache_root).resolve() / dataset_id
        obs_dir = ds_dir / "observations"
        if not obs_dir.is_dir() or not any(obs_dir.glob("*.parquet")):
            raise FileNotFoundError(
                f"Dataset {dataset_id!r} is not staged under {ds_dir}. "
                f"Run: uv run python scripts/stage_multimodal_datasets.py --dataset {dataset_id}"
            )
        return cls(spec=spec, dataset_dir=ds_dir)

    def load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.is_file():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"dataset_id": self.name}

    def create_backend(self) -> DuckDBBackend:
        """Return an `origin/main` `DuckDBBackend` locked strictly to `observations/`."""
        guard = SQLGuardConfig(
            allowed_data_dir=str(self.observations_dir),
            allowed_tables=frozenset({"observations"}),
        )
        return DuckDBBackend(str(self.dataset_dir), guard_config=guard)

    def search_sql(
        self,
        sql: str,
        *,
        requested_under: int = 0,
        hypothesis_family: str = "exploratory_sql",
        note_store: NoteStore | None = None,
    ) -> tuple[QueryExecutionResult, Note]:
        """Run a guarded read-only SQL magnet over `observations`, blocking label/file leaks."""
        if re.search(
            r"\b(read_json|read_csv|read_parquet|read_text|read_blob|ground_truth)\w*\b",
            sql,
            re.IGNORECASE,
        ):
            raise SQLValidationError(
                "Direct file-reader functions and ground_truth access are forbidden; "
                "query the registered `observations` table view instead."
            )
        backend = self.create_backend()
        qres, note = backend.execute_query(
            sql,
            requested_under=requested_under,
            hypothesis_family=hypothesis_family,
            note_store=note_store,
        )
        return qres, note

    def _summarize_primary_trigger(self, row: dict[str, Any]) -> str:
        if self.name == "caltech_camera_traps":
            return (
                f"MegaDetector conf={row.get('max_detection_conf', 0):.3f}, "
                f"detections={row.get('n_detections', 0)}, "
                f"max_bbox_area={row.get('max_bbox_area_frac', 0):.3f}"
            )
        if self.name == "btsbot":
            return (
                f"ZTF alert magpsf={row.get('magpsf', 0):.2f}±{row.get('sigmapsf', 0):.2f}, "
                f"deep_real_bogus drb={row.get('drb', 0):.4f}, dsnrms={row.get('dsnrms', 0):.1f}"
            )
        if self.name == "pyro_sdis":
            return (
                f"Optical gray_haze_fraction={row.get('gray_haze_fraction', 0):.3f}, "
                f"horizon_brightness={row.get('horizon_brightness', 0):.3f}"
            )
        return (
            f"Acoustic spike crest_factor={row.get('crest_factor_db', 0):.2f} dB, "
            f"rms={row.get('rms_db', 0):.1f} dBFS, peak={row.get('peak_amp', 0):.3f}"
        )

    def _summarize_contextual_telemetry(self, row: dict[str, Any]) -> str:
        if self.name == "caltech_camera_traps":
            return (
                f"Station={row.get('location_id')}, captured={row.get('captured_at')}, "
                f"hour_local={row.get('hour_local')}, burst_frames={row.get('seq_num_frames')}"
            )
        if self.name == "btsbot":
            return (
                f"prior_detections ndethist={row.get('ndethist')}, "
                f"host_star_galaxy_score sgscore1={row.get('sgscore1', 0):.3f}, "
                f"host_offset={row.get('distpsnr1', 0):.2f} arcsec, bad_pixels nbad={row.get('nbad')}"
            )
        if self.name == "pyro_sdis":
            return (
                f"Tower={row.get('partner')}/{row.get('camera')} (azimuth={row.get('azimuth_deg')}°), "
                f"sky_ground_contrast={row.get('sky_ground_contrast', 0):.3f}, "
                f"warm_glare={row.get('warm_glare_fraction', 0):.3f}"
            )
        return (
            f"duration={row.get('duration_s', 0):.2f}s, salience={row.get('salience')} "
            f"(1=foreground,2=background), centroid={row.get('spectral_centroid_hz', 0):.0f} Hz, "
            f"hf_energy_ratio={row.get('high_freq_energy_ratio', 0):.3f}"
        )

    def _ensure_candidate_media(
        self,
        row: dict[str, Any],
        *,
        fetch_on_demand: bool = True,
    ) -> ArtifactRef | None:
        record_id = str(row["record_id"])
        relpath = str(row.get("media_relpath") or "")
        local_path: Path | None = None
        if relpath:
            candidate_path = verify_safe_output_path(self.dataset_dir / relpath, self.dataset_dir)
            if candidate_path.is_file():
                local_path = candidate_path

        if local_path is None and fetch_on_demand:
            media_url = str(row.get("media_url") or "")
            if media_url:
                self.media_dir.mkdir(parents=True, exist_ok=True)
                ext = Path(urllib.parse.urlparse(media_url).path).suffix.lower() or ".jpg"
                target = verify_safe_output_path(
                    self.media_dir / f"{record_id}{ext}",
                    self.dataset_dir,
                )
                raw = safe_https_get_bytes(media_url)
                verify_magic_bytes(raw, ext)
                target.write_bytes(raw)
                local_path = target

        if local_path is None or not local_path.is_file():
            return None

        raw_bytes = local_path.read_bytes()
        ext = local_path.suffix.lower()
        verify_magic_bytes(raw_bytes, ext)
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        kind: ArtifactKind = (
            "plot_png" if self.name in ("urbansound8k", "btsbot") else "image"
        )
        return ArtifactRef(
            artifact_id=f"art_{self.name}_{record_id}",
            kind=kind,
            path=str(local_path),
            sha256=sha256_bytes(raw_bytes),
            mime_type=mime_type,
            produced_by_verifier=True,
        )

    def register_verifiers(
        self,
        registry: VerifierRegistry,
        *,
        fetch_on_demand: bool = True,
    ) -> None:
        """Register 3 deterministic TRUSTED verifiers for parallel ARES `solve()` execution."""

        def _fetch_row(
            con: duckdb.DuckDBPyConnection, params: dict[str, Any] | None
        ) -> dict[str, Any] | None:
            rid = (params or {}).get("record_id")
            if rid is None:
                return None
            cur = con.execute(
                "SELECT * FROM observations WHERE record_id = ? LIMIT 1",
                [str(rid)],
            )
            fetched = cur.fetchone()
            if fetched is None:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, fetched))

        def verify_primary_sensor_trigger(
            con: duckdb.DuckDBPyConnection,
            data_dir: str,
            params: dict[str, Any] | None = None,
        ) -> VerifierResult:
            row = _fetch_row(con, params)
            if row is None:
                return VerifierResult(
                    verifier_name="verify_primary_sensor_trigger",
                    passed=False,
                    certification="UNCERTIFIED",
                    evidence={"found": False},
                    row_count=0,
                    summary="Record not found.",
                )
            trig = self._summarize_primary_trigger(row)
            return VerifierResult(
                verifier_name="verify_primary_sensor_trigger",
                passed=True,
                certification="CERTIFIED",
                evidence={"record_id": str(row["record_id"]), "trigger_summary": trig},
                row_count=1,
                summary=trig,
            )

        def verify_contextual_telemetry(
            con: duckdb.DuckDBPyConnection,
            data_dir: str,
            params: dict[str, Any] | None = None,
        ) -> VerifierResult:
            row = _fetch_row(con, params)
            if row is None:
                return VerifierResult(
                    verifier_name="verify_contextual_telemetry",
                    passed=False,
                    certification="UNCERTIFIED",
                    evidence={"found": False},
                    row_count=0,
                    summary="Record not found.",
                )
            ctx = self._summarize_contextual_telemetry(row)
            return VerifierResult(
                verifier_name="verify_contextual_telemetry",
                passed=True,
                certification="CERTIFIED",
                evidence={"record_id": str(row["record_id"]), "context_summary": ctx},
                row_count=1,
                summary=ctx,
            )

        def verify_candidate_media(
            con: duckdb.DuckDBPyConnection,
            data_dir: str,
            params: dict[str, Any] | None = None,
        ) -> VerifierResult:
            row = _fetch_row(con, params)
            if row is None:
                return VerifierResult(
                    verifier_name="verify_candidate_media",
                    passed=False,
                    certification="UNCERTIFIED",
                    evidence={"found": False},
                    row_count=0,
                    summary="Record not found.",
                )
            art = self._ensure_candidate_media(row, fetch_on_demand=fetch_on_demand)
            rid = str(row["record_id"])
            if art is None:
                return VerifierResult(
                    verifier_name="verify_candidate_media",
                    passed=False,
                    certification="UNCERTIFIED",
                    evidence={"record_id": rid, "has_verified_media": False},
                    row_count=1,
                    summary=f"No visual artifact available for {rid}.",
                )
            media_desc = (
                f"Verified {art.mime_type} ({Path(art.path).name}, sha256={art.sha256[:12]}…)"
            )
            return VerifierResult(
                verifier_name="verify_candidate_media",
                passed=True,
                certification="CERTIFIED",
                evidence={
                    "record_id": rid,
                    "has_verified_media": True,
                    "artifact_id": art.artifact_id,
                    "media_summary": media_desc,
                },
                artifacts=[art],
                row_count=1,
                summary=media_desc,
            )

        registry.register(
            "verify_primary_sensor_trigger",
            verify_primary_sensor_trigger,
            description="Certify primary sensor/alert trigger threshold.",
            trusted=True,
        )
        registry.register(
            "verify_contextual_telemetry",
            verify_contextual_telemetry,
            description="Certify secondary contextual metadata columns.",
            trusted=True,
        )
        registry.register(
            "verify_candidate_media",
            verify_candidate_media,
            description="Attach SHA-256 verified visual media ArtifactRef for candidate.",
            trusted=True,
        )

    def run_ares_candidate_verifiers(
        self,
        record_id: str,
        *,
        note_store: NoteStore | None = None,
        fetch_on_demand: bool = True,
    ) -> tuple[NoteStore, list[ArtifactRef]]:
        """Dispatch all 3 TRUSTED verifiers in parallel through `ares.solve()`!"""
        store = note_store if note_store is not None else NoteStore()
        registry = InMemoryVerifierRegistry()
        self.register_verifiers(registry, fetch_on_demand=fetch_on_demand)
        backend = self.create_backend()
        graph = TaskGraph()

        tasks = [
            TaskSpec(
                task_id=f"t_trig_{record_id}",
                task_type="verify",
                verifier_name="verify_primary_sensor_trigger",
                verifier_params={"record_id": str(record_id)},
                hypothesis_family="primary_trigger",
                requested_under=0,
            ),
            TaskSpec(
                task_id=f"t_ctx_{record_id}",
                task_type="verify",
                verifier_name="verify_contextual_telemetry",
                verifier_params={"record_id": str(record_id)},
                hypothesis_family="contextual_telemetry",
                requested_under=0,
            ),
            TaskSpec(
                task_id=f"t_med_{record_id}",
                task_type="verify",
                verifier_name="verify_candidate_media",
                verifier_params={"record_id": str(record_id)},
                hypothesis_family="visual_media",
                requested_under=0,
            ),
        ]
        solve(
            tasks=tasks,
            graph=graph,
            backend=backend,
            registry=registry,
            note_store=store,
            max_concurrent=3,
        )
        arts = [
            art
            for n in store.snapshot()
            for art in n.artifacts
            if art.produced_by_verifier and Path(art.path).is_file()
        ]
        return store, arts

    def get_observation_row(self, record_id: str) -> dict[str, Any] | None:
        qres, _ = self.search_sql(
            f"SELECT * FROM observations WHERE record_id = '{str(record_id).replace(chr(39), chr(39)*2)}' LIMIT 1"
        )
        return qres.rows[0] if qres.rows else None

    def pick_duel_pair(self) -> tuple[str, str]:
        """Select 1 True Positive + 1 High-Salience Look-Alike Decoy with cached media."""
        gt = self.load_ground_truth()
        if self.name == "caltech_camera_traps":
            order_sql = "SELECT record_id FROM observations WHERE media_relpath != '' ORDER BY max_detection_conf DESC"
        elif self.name == "btsbot":
            order_sql = "SELECT record_id FROM observations WHERE media_relpath != '' ORDER BY drb DESC, magpsf ASC"
        elif self.name == "pyro_sdis":
            order_sql = "SELECT record_id FROM observations WHERE media_relpath != '' ORDER BY gray_haze_fraction DESC"
        else:
            order_sql = "SELECT record_id FROM observations WHERE media_relpath != '' ORDER BY crest_factor_db DESC"

        qres, _ = self.search_sql(order_sql)
        pos_id: str | None = None
        decoy_id: str | None = None
        for r in qres.rows:
            rid = str(r["record_id"])
            info = gt.get(rid, {})
            if info.get("is_true_event") and pos_id is None:
                pos_id = rid
            elif (not info.get("is_true_event")) and decoy_id is None:
                decoy_id = rid
            if pos_id and decoy_id:
                break
        if not pos_id or not decoy_id:
            raise RuntimeError(f"Could not find both positive & decoy cached rows in {self.name}")
        return pos_id, decoy_id

    def load_ground_truth(self) -> dict[str, dict[str, Any]]:
        if not self.ground_truth_path.is_file():
            raise FileNotFoundError(f"Ground truth file missing: {self.ground_truth_path}")
        raw = self.ground_truth_path.read_bytes()
        verify_magic_bytes(raw, ".json", max_bytes=DEFAULT_MAX_ARCHIVE_UNCOMPRESSED_BYTES)
        return dict(json.loads(raw.decode("utf-8"))["records"])
