"""Safe, zero-friction real-world multimodal datasets for ARES search & probing.

Bridges curated real-world multimodal datasets (Parquet observations + verified
`.png` / `.jpg` / `.wav` media + withheld ground-truth labels) directly into the
existing ARES stack (`DuckDBTrustedRunner`, `VerifierRegistry`, `NoteStore`,
`GeminiModelProvider`, and `DecisionProbeClient`).

Safety guarantees:
- Strict HTTPS host allowlist (`huggingface.co` + HF CDN, `storage.googleapis.com`,
  `lila.science`, `lilawildlife.blob.core.windows.net`).
- Safe file extensions ONLY (`.parquet`, `.csv`, `.json`, `.png`, `.jpg`, `.wav`).
  Explicitly blocks `.pkl`, `.pt`, `.bin`, `.sh`, and any dynamic code loading.
- Magic-byte signature verification and byte-size caps on every downloaded asset.
- Ground-truth labels are stored outside the SQL-exposed `observations/` directory
  (`ground_truth.json`) so agent SQL queries never leak labels.
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

from ares.duckdb_runner import DuckDBTrustedRunner, QueryExecutionResult
from ares.sql_guard import SQLGuardConfig, SQLValidationError
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

DEFAULT_MAX_MEDIA_BYTES: int = 25 * 1024 * 1024  # 25 MB per single image/audio clip
DEFAULT_MAX_ARCHIVE_UNCOMPRESSED_BYTES: int = 220 * 1024 * 1024  # 220 MB JSON metadata ceiling


class SafetyViolationError(ValueError):
    """Raised when a URL, path, archive member, or file header violates safety rules."""


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def verify_safe_https_url(url: str) -> str:
    """Validate that `url` uses HTTPS and points to an explicitly trusted host."""
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
            f"Refusing to write outside cache root {resolved_root}: {resolved_path}"
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
    """Verify magic-byte signatures and size bounds before persisting downloaded media."""
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
                        raise SafetyViolationError(f"Redirect missing Location header from {current_url}")
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
                    f"Uncompressed JSON size {info.file_size} exceeds safety cap {max_uncompressed_bytes}."
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
class DatasetSpec:
    """Static metadata describing one curated multimodal dataset."""

    dataset_id: str
    title: str
    domain: str
    source_url: str
    primary_media_kind: str
    positive_label_description: str
    decoy_description: str
    id_column: str
    schema_summary: str
    hypothesis_families: tuple[str, ...]
    decision_questions: tuple[DecisionQuestion, ...]


DATASET_CATALOG: dict[str, DatasetSpec] = {
    "btsbot": DatasetSpec(
        dataset_id="btsbot",
        title="ZTF BTSbot Supernova Transient Alerts (MultimodalUniverse/btsbot)",
        domain="Astronomy / Time-Domain Sky Survey",
        source_url="https://huggingface.co/datasets/MultimodalUniverse/btsbot",
        primary_media_kind="image/png (3-panel Science / Reference / Difference sky cutouts)",
        positive_label_description="Spectroscopically confirmed supernova (is_true_event=True)",
        decoy_description=(
            "High-SNR alert decoys: recurring variable stars, active galactic nuclei (AGN) "
            "nuclear flares, cosmic rays, and CCD image-subtraction ghosts"
        ),
        id_column="record_id",
        schema_summary=(
            "Table `observations` (1 row per Zwicky Transient Facility alert):\n"
            "  - record_id (VARCHAR): Unique alert identifier\n"
            "  - object_id (VARCHAR): ZTF sky object identifier\n"
            "  - jd (DOUBLE): Julian Date of observation\n"
            "  - magpsf, sigmapsf, diffmaglim (DOUBLE): PSF magnitude, uncertainty, and 5-sigma limit\n"
            "  - dsnrms, ssnrms (DOUBLE): Difference vs. science SNR metrics\n"
            "  - drb, new_drb (DOUBLE): Deep real-bogus alert score in [0, 1] (often >0.99 on decoys!)\n"
            "  - fwhm, chipsf, nbad, nneg, mindtoedge (DOUBLE/BIGINT): PSF width & CCD subtraction quality\n"
            "  - sgscore1, distpsnr1 (DOUBLE): Host star-galaxy score (0=galaxy, 1=star) & angular offset (arcsec)\n"
            "  - ndethist, ncovhist, age, days_since_peak (BIGINT/DOUBLE): Light-curve detection history\n"
            "  - acai_h, acai_v, acai_o, acai_n, acai_b (DOUBLE): Alert heuristic sub-scores\n"
            "  - is_rise, near_threshold (BOOLEAN): Photometric rise & threshold flags\n"
            "  - media_relpath (VARCHAR): Local relative path to 3-panel Science/Ref/Diff PNG stamp"
        ),
        hypothesis_families=(
            "extragalactic_rising_transient",
            "stellar_or_nuclear_agn_decoy",
            "ccd_subtraction_artifact_check",
            "host_galaxy_offset_verification",
        ),
        decision_questions=(
            DecisionQuestion(
                question_id="transient_classification",
                question="Based on the tabular alert metrics and Science/Reference/Difference sky cutouts, what is this candidate?",
                options=["real_supernova", "variable_or_agn_decoy", "subtraction_artifact", "uncertain"],
                question_type="multiple_choice",
            ),
            DecisionQuestion(
                question_id="trigger_spectroscopic_followup",
                question="Should scarce spectroscopic telescope follow-up be triggered for this alert?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ),
    ),
    "caltech_camera_traps": DatasetSpec(
        dataset_id="caltech_camera_traps",
        title="Caltech Camera Traps + MegaDetector v5 (LILA BC Open Data)",
        domain="Wildlife Ecology & Motion-Triggered Camera Traps",
        source_url="https://lila.science/datasets/caltech-camera-traps/",
        primary_media_kind="image/jpeg (Daytime RGB & Night IR camera-trap photographs)",
        positive_label_description="Real wild animal present in camera frame (is_true_event=True)",
        decoy_description=(
            "False motion triggers (`empty` frame): sun-heated swaying brush, wind gusts, "
            "shifting rock shadows, and IR glare that score high on automated motion/MegaDetector metrics"
        ),
        id_column="record_id",
        schema_summary=(
            "Table `observations` (243,100 motion-triggered Southwest US camera-trap captures):\n"
            "  - record_id (VARCHAR): Unique image capture ID\n"
            "  - location_id (VARCHAR): Camera station ID (140 stations)\n"
            "  - captured_at (VARCHAR): Local capture timestamp ('YYYY-MM-DD HH:MM:SS')\n"
            "  - hour_local (INTEGER): Hour of day (0-23; nocturnal vs. midday thermal wind)\n"
            "  - seq_id (VARCHAR): Multi-frame motion burst sequence ID\n"
            "  - seq_num_frames, frame_num (INTEGER): Total frames in burst & frame index\n"
            "  - width, height (INTEGER): Sensor frame dimensions\n"
            "  - max_detection_conf (DOUBLE): Automated MegaDetector v5a max confidence in [0, 1]\n"
            "  - n_detections (INTEGER): Number of detector boxes above 0.20 confidence\n"
            "  - max_bbox_area_frac (DOUBLE): Largest bounding box fraction of frame area\n"
            "  - media_url (VARCHAR): Trusted HTTPS URL to fetch full `.jpg` frame on demand\n"
            "  - media_relpath (VARCHAR): Local cached `.jpg` relative path (populated when cached)"
        ),
        hypothesis_families=(
            "high_conf_predator_sequence",
            "thermal_wind_brush_decoy",
            "nocturnal_wildlife_corridor",
            "single_frame_glare_or_shadow",
        ),
        decision_questions=(
            DecisionQuestion(
                question_id="camera_trigger_cause",
                question="What caused this motion-triggered camera-trap capture?",
                options=["real_wildlife", "empty_false_trigger", "human_or_vehicle", "uncertain"],
                question_type="multiple_choice",
            ),
            DecisionQuestion(
                question_id="dispatch_field_biologist_alert",
                question="Is a real target animal visibly confirmed in the camera-trap photograph?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ),
    ),
    "pyro_sdis": DatasetSpec(
        dataset_id="pyro_sdis",
        title="Pyronear Early Wildfire Lookout Network (pyronear/pyro-sdis)",
        domain="Environmental Monitoring / Wildfire Smoke Detection",
        source_url="https://huggingface.co/datasets/pyronear/pyro-sdis",
        primary_media_kind="image/jpeg (Mountaintop fire-lookout tower photographs)",
        positive_label_description="Confirmed early wildfire smoke plume (is_true_event=True)",
        decoy_description=(
            "Hard-negative lookout false alarms: low valley fog banks, cumulus cloud bases, "
            "harvest dust, and afternoon lens flare triggering tower optical alerts"
        ),
        id_column="record_id",
        schema_summary=(
            "Table `observations` (1,129 mountaintop fire-lookout tower alert frames):\n"
            "  - record_id (VARCHAR): Unique tower capture identifier\n"
            "  - partner (VARCHAR): Fire network operator ('biobio', 'engie', 'sdis', etc.)\n"
            "  - camera (VARCHAR): Tower site & azimuth heading (e.g. 'florida-180')\n"
            "  - azimuth_deg (INTEGER): Camera compass bearing when encoded in tower ID\n"
            "  - captured_at (VARCHAR): ISO-like capture timestamp\n"
            "  - hour_obs (INTEGER): Capture hour (0-23)\n"
            "  - width, height, jpg_bytes (INTEGER): Frame dimensions & compressed size\n"
            "  - brightness_mean, brightness_std (DOUBLE): Full-frame luminance statistics\n"
            "  - horizon_brightness, sky_ground_contrast (DOUBLE): Upper-vs-lower band luminance metrics\n"
            "  - gray_haze_fraction (DOUBLE): Fraction of low-saturation bright haze pixels (elevated on BOTH smoke and fog decoys!)\n"
            "  - warm_glare_fraction (DOUBLE): Fraction of warm sun-glare pixels\n"
            "  - media_relpath (VARCHAR): Local relative path to verified lookout `.jpg` frame"
        ),
        hypothesis_families=(
            "localized_rising_smoke_plume",
            "valley_fog_or_cloud_deck_decoy",
            "solar_glare_optical_false_alarm",
            "multi_frame_tower_persistence",
        ),
        decision_questions=(
            DecisionQuestion(
                question_id="lookout_alert_assessment",
                question="What does the mountaintop lookout tower photograph show?",
                options=["wildfire_smoke_plume", "cloud_or_fog_decoy", "glare_or_haze_decoy", "uncertain"],
                question_type="multiple_choice",
            ),
            DecisionQuestion(
                question_id="dispatch_fire_suppression_check",
                question="Is a genuine smoke plume visible that warrants fire-dispatch verification?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ),
    ),
    "urbansound8k": DatasetSpec(
        dataset_id="urbansound8k",
        title="UrbanSound8K Acoustic Sensor Network (danavery/urbansound8K)",
        domain="Urban Acoustic Sensors + Field Audio (.wav)",
        source_url="https://huggingface.co/datasets/danavery/urbansound8K",
        primary_media_kind="audio/wav + deterministic waveform/spectrogram PNG card",
        positive_label_description="High-priority emergency acoustic event: gun_shot or siren (is_true_event=True)",
        decoy_description=(
            "Impulsive urban noise decoys: construction jackhammers, car horns, dog barks, "
            "and street clatter whose peak-to-RMS crest factor mimics gunfire or sirens in tabular telemetry"
        ),
        id_column="record_id",
        schema_summary=(
            "Table `observations` (Urban acoustic microphone trigger slices):\n"
            "  - record_id (VARCHAR): Unique audio slice identifier\n"
            "  - slice_file_name (VARCHAR): Original `.wav` slice filename\n"
            "  - freesound_id (BIGINT): Source field recording ID\n"
            "  - start_s, end_s, duration_s (DOUBLE): Clip window & duration in seconds\n"
            "  - salience (BIGINT): Sensor foreground=1 vs. noisy background=2 estimate\n"
            "  - fold (BIGINT): Acoustic station fold (1-10)\n"
            "  - sample_rate_hz, channels (INTEGER): Audio recording format\n"
            "  - rms_db (DOUBLE): Root-mean-square acoustic energy (dBFS)\n"
            "  - peak_amp (DOUBLE): Normalized peak waveform amplitude in [0, 1]\n"
            "  - crest_factor_db (DOUBLE): Peak-to-RMS impulsive spike ratio in dB (high on gunshots AND jackhammer/bark decoys!)\n"
            "  - zero_crossing_rate (DOUBLE): Waveform zero-crossing rate\n"
            "  - spectral_centroid_hz (DOUBLE): FFT magnitude-weighted mean frequency (Hz)\n"
            "  - high_freq_energy_ratio (DOUBLE): Energy fraction above 2.5 kHz\n"
            "  - wav_relpath (VARCHAR): Local relative path to verified `.wav` audio clip\n"
            "  - media_relpath (VARCHAR): Local relative path to deterministic waveform+spectrogram `.png`"
        ),
        hypothesis_families=(
            "impulsive_ballistic_transient",
            "harmonic_emergency_siren",
            "construction_or_traffic_decoy",
            "background_clutter_false_trigger",
        ),
        decision_questions=(
            DecisionQuestion(
                question_id="acoustic_event_type",
                question="Based on the acoustic sensor metrics and spectrogram/waveform card, what produced this sound?",
                options=["emergency_gunshot_or_siren", "construction_or_horn_decoy", "ambient_urban_noise", "uncertain"],
                question_type="multiple_choice",
            ),
            DecisionQuestion(
                question_id="escalate_acoustic_alert",
                question="Is this a genuine emergency acoustic signature (gunshot or siren) rather than urban noise?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ),
    ),
}


class MultimodalSearchDataset:
    """Drop-in adapter making staged `.data_cache/<dataset_id>` immediately searchable in ARES.

    Example (3 lines):
        ds = MultimodalSearchDataset.open("btsbot")
        rows, sql_note = ds.search_sql("SELECT record_id, magpsf, drb, ndethist FROM observations WHERE drb > 0.99 LIMIT 5")
        ver_res, img_note = ds.inspect_candidate(rows[0]["record_id"])
    """

    def __init__(self, spec: DatasetSpec, dataset_dir: Path) -> None:
        self.spec = spec
        self.name = spec.dataset_id
        self.dataset_dir = dataset_dir.resolve()
        self.observations_dir = self.dataset_dir / "observations"
        self.media_dir = self.dataset_dir / "media"
        self.ground_truth_path = self.dataset_dir / "ground_truth.json"
        self.manifest_path = self.dataset_dir / "manifest.json"

    @classmethod
    def open(
        cls,
        dataset_id: str,
        *,
        cache_root: str | Path = ".data_cache",
    ) -> "MultimodalSearchDataset":
        """Open a staged multimodal dataset from `<cache_root>/<dataset_id>`."""
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

    def schema_description(self) -> str:
        """Table & column semantics ready for `GeminiModelProvider` prompts."""
        return (
            f"Dataset: {self.spec.title} ({self.spec.domain})\n"
            f"Primary Media: {self.spec.primary_media_kind}\n"
            f"Decoy Warning: {self.spec.decoy_description}\n\n"
            f"{self.spec.schema_summary}\n"
        )

    def decision_questions(self) -> list[DecisionQuestion]:
        """Closed-set questions ready for `DecisionProbeClient` (`gemini-decision-1`)."""
        return list(self.spec.decision_questions)

    def hypothesis_families(self) -> list[str]:
        """Parallel swarm exploration families ready for `ARESRunner`."""
        return list(self.spec.hypothesis_families)

    def create_duckdb_runner(self) -> DuckDBTrustedRunner:
        """Create an ARES `DuckDBTrustedRunner` locked strictly to `observations/`."""
        guard = SQLGuardConfig(
            allowed_data_dir=str(self.observations_dir),
            allowed_tables=frozenset({"observations"}),
        )
        return DuckDBTrustedRunner(str(self.dataset_dir), guard_config=guard)

    def search_sql(
        self,
        sql: str,
        *,
        requested_under: int = 0,
        hypothesis_family: str = "exploratory_sql",
        note_store: NoteStore | None = None,
    ) -> tuple[list[dict[str, Any]], Note]:
        """Run a guarded read-only SQL query over `observations` via `DuckDBTrustedRunner`."""
        if re.search(
            r"\b(read_json|read_csv|read_parquet|read_text|read_blob|ground_truth)\w*\b",
            sql,
            re.IGNORECASE,
        ):
            raise SQLValidationError(
                "Direct file-reader functions and ground_truth access are forbidden; "
                "query the registered `observations` table view instead."
            )
        runner = self.create_duckdb_runner()
        qres: QueryExecutionResult
        qres, note = runner.execute_readonly_query(
            sql,
            requested_under=requested_under,
            hypothesis_family=hypothesis_family,
            note_store=note_store,
        )
        return qres.rows, note

    def _ensure_candidate_media(
        self,
        row: dict[str, Any],
        *,
        fetch_on_demand: bool = True,
    ) -> ArtifactRef | None:
        """Resolve (or safely fetch on-demand) the visual `.png`/`.jpg` artifact for `row`."""
        record_id = str(row["record_id"])
        relpath = row.get("media_relpath") or ""
        local_path: Path | None = None
        if relpath:
            candidate_path = verify_safe_output_path(self.dataset_dir / relpath, self.dataset_dir)
            if candidate_path.is_file():
                local_path = candidate_path

        if local_path is None and fetch_on_demand:
            media_url = row.get("media_url") or ""
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
        kind: ArtifactKind = "plot_png" if self.name == "urbansound8k" else "image"
        return ArtifactRef(
            artifact_id=f"art_{self.name}_{record_id}",
            kind=kind,
            path=str(local_path),
            sha256=sha256_bytes(raw_bytes),
            mime_type=mime_type,
            produced_by_verifier=True,
        )

    def inspect_candidate(
        self,
        record_id: str,
        *,
        requested_under: int = 0,
        note_store: NoteStore | None = None,
        fetch_on_demand: bool = True,
    ) -> tuple[VerifierResult, Note]:
        """Run deterministic TRUSTED candidate lookup + attach verified media `ArtifactRef`."""
        registry = InMemoryVerifierRegistry()
        self.register_verifiers(registry, fetch_on_demand=fetch_on_demand)
        runner = self.create_duckdb_runner()
        safe_id = str(record_id).replace("'", "''")
        return runner.execute_verifier(
            registry,
            "verify_candidate_media",
            params={"record_id": str(record_id)},
            query_sql=f"SELECT * FROM observations WHERE record_id = '{safe_id}' LIMIT 1",
            requested_under=requested_under,
            hypothesis_family="candidate_media_inspection",
            note_store=note_store,
        )

    def register_verifiers(
        self,
        registry: VerifierRegistry,
        *,
        fetch_on_demand: bool = True,
    ) -> None:
        """Register deterministic `TRUSTED` DuckDB + media verifiers for ARES."""

        def verify_candidate_media(
            con: duckdb.DuckDBPyConnection,
            data_dir: str,
            params: dict[str, Any] | None = None,
        ) -> VerifierResult:
            effective_params = params or {}
            record_id = effective_params.get("record_id")
            row: dict[str, Any] | None = None
            if record_id is not None:
                cur = con.execute(
                    "SELECT * FROM observations WHERE record_id = ? LIMIT 1",
                    [str(record_id)],
                )
                fetched = cur.fetchone()
                if fetched is not None:
                    cols = [d[0] for d in cur.description]
                    row = dict(zip(cols, fetched))
            else:
                cur = con.execute(
                    "SELECT * FROM observations WHERE media_relpath != '' LIMIT 1"
                )
                fetched = cur.fetchone()
                if fetched is not None:
                    cols = [d[0] for d in cur.description]
                    row = dict(zip(cols, fetched))

            if row is None:
                return VerifierResult(
                    verifier_name="verify_candidate_media",
                    passed=False,
                    certification="UNCERTIFIED",
                    evidence={"dataset": self.name, "found": False},
                    row_count=0,
                    summary=f"No candidate record found in {self.name}.observations.",
                )

            art = self._ensure_candidate_media(row, fetch_on_demand=fetch_on_demand)
            artifacts = [art] if art is not None else []
            clean_row = {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in row.items()
            }
            rid = str(row["record_id"])
            passed = art is not None
            return VerifierResult(
                verifier_name="verify_candidate_media",
                passed=passed,
                certification="CERTIFIED" if passed else "UNCERTIFIED",
                evidence={
                    "dataset": self.name,
                    "record_id": rid,
                    "observation": clean_row,
                    "has_verified_media": passed,
                    "artifact_id": art.artifact_id if art else None,
                },
                artifacts=artifacts,
                row_count=1,
                summary=(
                    f"Verified observation {rid} in {self.name} "
                    f"({'attached ' + art.mime_type if art else 'tabular only'})."
                ),
            )

        registry.register(
            "verify_candidate_media",
            verify_candidate_media,
            description=(
                f"Deterministically fetch observation row and verified media ArtifactRef "
                f"for {self.spec.title}."
            ),
            trusted=True,
        )

    def load_ground_truth(self) -> dict[str, dict[str, Any]]:
        """Load withheld ground-truth labels (`record_id -> truth dict`) for scoring ONLY."""
        if not self.ground_truth_path.is_file():
            raise FileNotFoundError(f"Ground truth file missing: {self.ground_truth_path}")
        raw = self.ground_truth_path.read_bytes()
        verify_magic_bytes(raw, ".json", max_bytes=DEFAULT_MAX_ARCHIVE_UNCOMPRESSED_BYTES)
        data = json.loads(raw.decode("utf-8"))
        return dict(data["records"])

    def score_predictions(self, predictions: dict[str, bool]) -> dict[str, Any]:
        """Score `record_id -> predicted_is_true_event` against withheld ground truth."""
        truth = self.load_ground_truth()
        evaluated = [rid for rid in predictions if rid in truth]
        if not evaluated:
            return {
                "evaluated_count": 0,
                "accuracy": 0.0,
                "decoy_rejection_rate": 0.0,
            }
        tp = fp = tn = fn = 0
        decoy_total = decoy_rejected = 0
        for rid in evaluated:
            pred = bool(predictions[rid])
            gt_item = truth[rid]
            actual = bool(gt_item["is_true_event"])
            is_decoy = bool(gt_item.get("is_decoy", not actual))
            if actual and pred:
                tp += 1
            elif not actual and pred:
                fp += 1
            elif not actual and not pred:
                tn += 1
            else:
                fn += 1
            if is_decoy and not actual:
                decoy_total += 1
                if not pred:
                    decoy_rejected += 1

        n = len(evaluated)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return {
            "dataset": self.name,
            "evaluated_count": n,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": round((tp + tn) / n, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "decoy_count": decoy_total,
            "decoy_rejection_rate": (
                round(decoy_rejected / decoy_total, 4) if decoy_total else 1.0
            ),
        }


__all__ = [
    "ALLOWED_HTTPS_HOSTS",
    "ALLOWED_SAFE_EXTENSIONS",
    "DATASET_CATALOG",
    "DatasetSpec",
    "FORBIDDEN_EXTENSIONS",
    "MultimodalSearchDataset",
    "SafetyViolationError",
    "safe_extract_single_json_from_zip",
    "safe_https_get_bytes",
    "sha256_bytes",
    "verify_magic_bytes",
    "verify_safe_https_url",
    "verify_safe_output_path",
]
