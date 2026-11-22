"""End-to-end Home Inventory Ledger orchestrator (`ledger/loop.py`).

Pipeline:
  1. Ingest (deterministic): ffmpeg keyframes + dHash dedup + Laplacian blur/brightness + WAV audio
  2. Understand (`gemini-4-carbon`): room segments, timestamped spoken claims, Coverage C policy limit
  3. Extract (`gemini-4-hydrogen` parallel workers): ItemObservation[] + catalog search magnets
  4. Resolve (`ledger/resolve.py`): cross-angle dedup + spoken claim alignment + contradiction flags
  5. Price & Verify (`ShopMCPClient` + `gemini-decision-1` match + 9 deterministic verifiers):
     produces immutable `Note`s -> `apply_patch` enforcing Invariants I1-I7 + `rebuild_board` 0-drift check
  6. Decide, Probe & Ask (`gemini-decision-1` + `ledger/voi.py`): coverage verdict, counterfactual TV
     influence, and ranked "show me X" capture requests with dollar VoI
  7. Incremental Follow-up (`run_followup`): processes a 5s close-up clip or spoken answer,
     flips target item from UNCERTIFIED -> CERTIFIED when `catalog_exact_match` passes, and re-runs verdict!
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

import numpy as np

from commerce.shop_mcp import ShopMCPClient
from contracts import (
    BoardPatch,
    ItemPatch,
    LedgerBoard,
    Listing,
    Note,
    PipelineTelemetry,
    PolicyInfo,
)
from ingest.audio import extract_audio_wav
from ingest.frames import crop_bbox, extract_frames
from ledger.board import (
    append_notes_and_listings,
    apply_patch,
    compute_board_drift,
    rebuild_board,
)
from ledger.resolve import ResolvedItemTrack, resolve_observations
from ledger.verifiers import (
    catalog_exact_match,
    frame_quality,
    listing_freshness,
    model_number_format,
    policy_limit_parse,
    price_sanity,
    quantity_consistency,
    room_coverage,
)
from ledger.voi import APPRAISAL_CATEGORIES, rank_capture_requests
from models.carbon import CarbonClient
from models.decision import DecisionClient
from models.hydrogen import HydrogenClient
from models.provider import GenAIRuntime

ProgressCallback = Callable[[str, str, dict], None]


def _noop_progress(stage: str, message: str, meta: dict) -> None:
    return None


def _percentile_range(prices: list[float]) -> tuple[float, float]:
    arr = np.asarray(sorted(prices), dtype=np.float64)
    if len(arr) == 1:
        v = float(arr[0])
        return round(v * 0.85, 2), round(v * 1.15, 2)
    lo = float(np.percentile(arr, 25))
    hi = float(np.percentile(arr, 75))
    if hi - lo < 1.0:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo:
        hi = round(lo * 1.20, 2)
    return round(lo, 2), round(hi, 2)


class InventoryLedgerPipeline:
    def __init__(
        self,
        work_dir: str | Path = "data/sessions",
        runtime: GenAIRuntime | None = None,
        shop_client: ShopMCPClient | None = None,
        max_concurrent_segments: int = 4,
        max_probe_items: int = 6,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.rt = runtime or GenAIRuntime()
        self.shop = shop_client or ShopMCPClient(replay_mode=self.rt.replay_mode)
        self.carbon = CarbonClient(self.rt)
        self.hydrogen = HydrogenClient(self.rt)
        self.decision = DecisionClient(self.rt)
        self.max_concurrent_segments = max_concurrent_segments
        self.max_probe_items = max_probe_items

    def _price_and_verify_track(
        self,
        track: ResolvedItemTrack,
        session_dir: Path,
        board_version: int,
        note_seq_start: int,
    ) -> tuple[list[Note], list[Listing], ItemPatch]:
        """Run Shop MCP search + Decision-1 listing filter + deterministic verifiers on one item track."""
        notes: list[Note] = []
        seq = note_seq_start

        def _next_nid(suffix: str) -> str:
            nonlocal seq
            seq += 1
            return f"note_{track.item_id}_{seq:03d}_{suffix}"

        # 1. Check frame quality + model_number_format for identity certification
        fq_ok, fq_det = frame_quality(track.best_frame)
        fmt_ok, fmt_det = model_number_format(track.model_text, track.brand)

        flags = list(track.flags)
        if not fq_ok:
            for r in fq_det.get("reasons", []):
                if r not in flags:
                    flags.append(r)

        frame_ids = [track.best_frame.frame_id] if track.best_frame else []
        crop_path: str | None = None
        if track.best_frame and Path(track.best_frame.path).exists():
            crop_path = crop_bbox(
                track.best_frame.path,
                track.best_obs.bbox if track.best_obs else None,
                session_dir / "crops" / f"{track.item_id}.jpg",
            )

        # Quantity consistency verifier
        qty_ok, _ = quantity_consistency(track.quantity, track.simultaneous_counts)
        verified_qty = track.quantity if qty_ok else max(track.simultaneous_counts or [1])
        notes.append(
            Note(
                note_id=_next_nid("qty"),
                item_id=track.item_id,
                certification="CERTIFIED" if qty_ok else "UNCERTIFIED",
                kind="quantity",
                verifier="quantity_consistency",
                frame_ids=frame_ids,
                value={"quantity": verified_qty},
                summary=f"Verified quantity={verified_qty} across {len(track.observations)} frame sightings",
                requested_under=board_version,
            )
        )

        # Contradiction note if narration conflicted with visual logo/label
        if track.narration_conflict:
            notes.append(
                Note(
                    note_id=_next_nid("conflict"),
                    item_id=track.item_id,
                    certification="UNCERTIFIED",
                    kind="contradiction",
                    verifier=None,
                    frame_ids=frame_ids,
                    claim_ids=[c.claim_id for c in track.matched_claims],
                    value={"flags": ["narration_conflict"]},
                    summary=track.narration_conflict,
                    requested_under=board_version,
                )
            )

        # 2. Search Shop MCP using Hydrogen's ranked search magnets
        seen_lids: dict[str, Listing] = {}
        for query in track.catalog_queries[:3]:
            found = self.shop.search_catalog(query, limit=6)
            for lst in found:
                if lst.listing_id not in seen_lids:
                    seen_lids[lst.listing_id] = lst
            if len(seen_lids) >= 8:
                break

        raw_candidates = list(seen_lids.values())

        # Filter candidates through pure-Python `price_sanity` + `listing_freshness`
        sane_candidates: list[Listing] = []
        for lst in raw_candidates:
            ps_ok, _ = price_sanity(lst, raw_candidates, item_category=track.category)
            lf_ok, _ = listing_freshness(lst)
            if ps_ok and lf_ok:
                sane_candidates.append(lst)

        # Filter through `gemini-decision-1` closed-set visual/product compatibility
        match_scores = self.decision.match_listings(
            item_name=track.display_name,
            item_category=track.category,
            item_description=track.description,
            frame_path=crop_path or (track.best_frame.path if track.best_frame else None),
            listings=sane_candidates,
        )
        matched_listings = [
            lst
            for lst in sane_candidates
            if match_scores.get(lst.listing_id, (0.0, 0.0))[0] >= 0.50
        ]

        # 3. Check deterministic `catalog_exact_match` -> CERTIFIED iff frame_quality + model_format + exact_match pass!
        certified_listing: Listing | None = None
        if fq_ok and fmt_ok:
            for lst in matched_listings:
                exact_ok, _ = catalog_exact_match(track.model_text, track.brand, lst)
                if exact_ok:
                    certified_listing = lst
                    break

        base_meta = {
            "room": track.room,
            "category": track.category,
            "display_name": track.display_name,
            "brand": track.brand,
            "model": fmt_det.get("normalized") if fmt_ok else track.model_text,
            "description": track.description,
            "best_frame_id": track.best_frame.frame_id if track.best_frame else None,
            "best_frame_path": crop_path or (track.best_frame.path if track.best_frame else None),
            "first_seen_ms": track.observations[0].ts_ms if track.observations else 0,
            "spoken_claim_text": track.matched_claims[0].text if track.matched_claims else None,
            "flags": flags,
        }

        if certified_listing is not None:
            id_note = Note(
                note_id=_next_nid("id_cert"),
                item_id=track.item_id,
                certification="CERTIFIED",
                kind="identity",
                verifier="model_number_format",
                frame_ids=frame_ids,
                value=base_meta,
                summary=f"Legible model '{track.model_text}' passed format + frame_quality gates",
                requested_under=board_version,
            )
            pr_note = Note(
                note_id=_next_nid("price_cert"),
                item_id=track.item_id,
                certification="CERTIFIED",
                kind="price",
                verifier="catalog_exact_match",
                frame_ids=frame_ids,
                listing_ids=[certified_listing.listing_id],
                value={"unit_price": certified_listing.price},
                summary=f"CERTIFIED against live listing '{certified_listing.title}' (${certified_listing.price:,.2f})",
                requested_under=board_version,
            )
            notes.extend([id_note, pr_note])
            patch = ItemPatch(
                item_id=track.item_id,
                room=track.room,
                category=track.category,
                display_name=track.display_name,
                brand=track.brand,
                model=fmt_det.get("normalized"),
                description=track.description,
                quantity=verified_qty,
                identity_status="CERTIFIED",
                price_status="CERTIFIED",
                unit_price=certified_listing.price,
                matched_listing=certified_listing,
                candidate_listings=matched_listings,
                best_frame_id=base_meta["best_frame_id"],
                best_frame_path=base_meta["best_frame_path"],
                first_seen_ms=base_meta["first_seen_ms"],
                spoken_claim_text=base_meta["spoken_claim_text"],
                conflict_detail=track.narration_conflict,
                cite_note_ids=[n.note_id for n in notes],
                add_flags=flags,
            )
            return notes, sane_candidates, patch

        if matched_listings and track.category not in APPRAISAL_CATEGORIES:
            lo, hi = _percentile_range([lst.price for lst in matched_listings])
            id_note = Note(
                note_id=_next_nid("id_uncert"),
                item_id=track.item_id,
                certification="UNCERTIFIED",
                kind="identity",
                verifier=None,
                frame_ids=frame_ids,
                value=base_meta,
                summary=f"Visual/spoken identification without verified exact catalog model match ({track.display_name})",
                requested_under=board_version,
            )
            pr_note = Note(
                note_id=_next_nid("price_range"),
                item_id=track.item_id,
                certification="UNCERTIFIED",
                kind="price",
                verifier="price_sanity",
                frame_ids=frame_ids,
                listing_ids=[lst.listing_id for lst in matched_listings],
                value={"price_low": lo, "price_high": hi},
                summary=f"UNCERTIFIED 25th-75th percentile range ${lo:,.0f}-${hi:,.0f} across {len(matched_listings)} listings",
                requested_under=board_version,
            )
            notes.extend([id_note, pr_note])
            patch = ItemPatch(
                item_id=track.item_id,
                room=track.room,
                category=track.category,
                display_name=track.display_name,
                brand=track.brand,
                model=track.model_text,
                description=track.description,
                quantity=verified_qty,
                identity_status="UNCERTIFIED",
                price_status="UNCERTIFIED",
                price_low=lo,
                price_high=hi,
                matched_listing=None,
                candidate_listings=matched_listings,
                best_frame_id=base_meta["best_frame_id"],
                best_frame_path=base_meta["best_frame_path"],
                first_seen_ms=base_meta["first_seen_ms"],
                spoken_claim_text=base_meta["spoken_claim_text"],
                conflict_detail=track.narration_conflict,
                cite_note_ids=[n.note_id for n in notes],
                add_flags=flags,
            )
            return notes, sane_candidates, patch

        # Honest UNKNOWN (no credible catalog coverage or requires human appraisal)
        if track.category in APPRAISAL_CATEGORIES and "needs_appraisal" not in flags:
            flags.append("needs_appraisal")
        elif "catalog_coverage_gap" not in flags:
            flags.append("catalog_coverage_gap")
        base_meta["flags"] = flags

        id_note = Note(
            note_id=_next_nid("id_unk"),
            item_id=track.item_id,
            certification="UNCERTIFIED",
            kind="identity",
            verifier=None,
            frame_ids=frame_ids,
            value=base_meta,
            summary=f"Seen in video ({track.display_name}), no verified Shop catalog match",
            requested_under=board_version,
        )
        notes.append(id_note)
        patch = ItemPatch(
            item_id=track.item_id,
            room=track.room,
            category=track.category,
            display_name=track.display_name,
            brand=track.brand,
            model=track.model_text,
            description=track.description,
            quantity=verified_qty,
            identity_status="UNCERTIFIED",
            price_status="UNKNOWN",
            best_frame_id=base_meta["best_frame_id"],
            best_frame_path=base_meta["best_frame_path"],
            first_seen_ms=base_meta["first_seen_ms"],
            spoken_claim_text=base_meta["spoken_claim_text"],
            conflict_detail=track.narration_conflict,
            cite_note_ids=[n.note_id for n in notes],
            add_flags=flags,
        )
        return notes, sane_candidates, patch

    def run_walkthrough(
        self,
        video_path: str | Path,
        session_id: str = "session_01",
        dec_page_path: str | Path | None = None,
        manual_coverage_limit: float | None = None,
        manual_deductible: float | None = None,
        expected_rooms: list[str] | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> LedgerBoard:
        """Execute full multimodal pipeline from walkthrough video + policy to coverage verdict & asks."""
        cb = progress_cb or _noop_progress
        t0 = time.perf_counter()
        sdir = self.work_dir / session_id
        sdir.mkdir(parents=True, exist_ok=True)

        # Step 1: Deterministic ingest
        cb("ingest", "Extracting keyframes, computing 64-bit dHash & Laplacian sharpness…", {})
        frames, raw_frame_count = extract_frames(video_path, sdir / "ingest", video_id=session_id)
        audio_wav = extract_audio_wav(video_path, sdir / "ingest" / "narration.wav")
        frames_by_id = {f.frame_id: f for f in frames}

        # Step 2: Gemini 4 Carbon full-video + audio + declarations page pass
        cb(
            "understand",
            f"Gemini 4 Carbon ({self.rt.carbon_model}) segmenting clip, aligning speech & parsing policy…",
            {"frames_kept": len(frames), "frames_raw": raw_frame_count},
        )
        segments, claims, extracted_policy = self.carbon.understand(
            video_path=video_path,
            frames=frames,
            audio_wav_path=audio_wav,
            dec_page_path=dec_page_path,
            video_id=session_id,
        )

        eff_limit = manual_coverage_limit if manual_coverage_limit is not None else extracted_policy.coverage_c_limit
        eff_ded = manual_deductible if manual_deductible is not None else extracted_policy.deductible
        pol_ok, _ = policy_limit_parse(eff_limit, eff_ded)
        policy = PolicyInfo(
            coverage_c_limit=eff_limit if pol_ok else None,
            deductible=eff_ded if pol_ok else None,
            carrier=extracted_policy.carrier,
            confidence=extracted_policy.confidence if pol_ok else 0.0,
            source_path=extracted_policy.source_path,
        )

        # Step 3: Parallel Gemini 4 Hydrogen extraction
        cb(
            "extract",
            f"Running {len(segments)} parallel Gemini 4 Hydrogen ({self.rt.hydrogen_model}) segment workers…",
            {"segments": len(segments), "claims": len(claims)},
        )
        observations = self.hydrogen.extract_segments_parallel(
            segments=segments,
            all_frames=frames,
            all_claims=claims,
            max_concurrent=self.max_concurrent_segments,
        )

        # Step 4: Deduplication + spoken contradiction detection
        cb("resolve", "Deduplicating multi-angle sightings & checking spoken vs. visual contradictions…", {})
        seg_rooms = {s.segment_id: s.room_label for s in segments}
        tracks = resolve_observations(
            observations=observations,
            frames_by_id=frames_by_id,
            segment_rooms=seg_rooms,
            claims=claims,
        )

        # Step 5: Shop MCP pricing + Decision-1 match + Deterministic Verifiers + Invariant Board
        cb(
            "price_verify",
            f"Querying Shop MCP & verifying {len(tracks)} distinct items via deterministic verifiers…",
            {"tracks": len(tracks)},
        )
        board = LedgerBoard(session_id=session_id, policy=policy)
        all_new_notes: list[Note] = []
        all_new_listings: list[Listing] = []
        item_patches: list[ItemPatch] = []

        unopened_spaces = [
            sp for s in segments for sp in s.closets_or_drawers_unopened
        ]
        _, rc_det = room_coverage(
            rooms_seen=[s.room_label for s in segments],
            expected_rooms=expected_rooms,
            unopened_spaces=unopened_spaces,
        )
        cov_note = Note(
            note_id=f"note_cov_{board.board_version}",
            item_id="__board__",
            certification="CERTIFIED",
            kind="coverage",
            verifier="room_coverage",
            value={
                "rooms_seen": rc_det["rooms_seen"],
                "rooms_missing": rc_det["rooms_missing"],
            },
            summary="Room and unopened closet coverage check",
            requested_under=board.board_version,
        )
        all_new_notes.append(cov_note)

        seq_counter = 0
        for tr in tracks:
            t_notes, t_listings, t_patch = self._price_and_verify_track(
                tr, sdir, board.board_version, seq_counter
            )
            seq_counter += len(t_notes)
            all_new_notes.extend(t_notes)
            all_new_listings.extend(t_listings)
            item_patches.append(t_patch)

        board = append_notes_and_listings(board, all_new_notes, all_new_listings)
        board = apply_patch(
            board,
            BoardPatch(
                requested_under=0,
                item_patches=item_patches,
                rooms_seen=rc_det["rooms_seen"],
                rooms_missing=rc_det["rooms_missing"],
            ),
        )

        # Invariant I3 check: rebuild board strictly from notes and verify 0 drift!
        rebuilt = rebuild_board(board.notes, board.listings_index, board.policy, session_id=session_id)
        drift = compute_board_drift(board, rebuilt)

        # Step 6: Gemini Decision-1 Verdict + Counterfactual Influence + VoI Capture Requests
        cb(
            "decide_probe",
            f"Gemini Decision-1 ({self.rt.decision_model}) computing coverage verdict & counterfactual VoI…",
            {},
        )
        probe_out = self.decision.decide_and_probe(board, max_probe_items=self.max_probe_items)
        ranked_items, asks = rank_capture_requests(board, probe_out.influences, max_requests=3)

        elapsed = round(time.perf_counter() - t0, 2)
        telemetry = PipelineTelemetry(
            carbon_calls=self.rt.carbon_calls,
            hydrogen_calls=self.rt.hydrogen_calls,
            decision_calls=self.rt.decision_calls,
            shop_mcp_calls=self.shop.call_count,
            frames_extracted=raw_frame_count,
            frames_kept_after_dedup=len(frames),
            board_drift_count=drift,
            rejected_patches=board.telemetry.rejected_patches,
            stale_notes_logged=board.telemetry.stale_notes_logged,
            elapsed_seconds=elapsed,
            mode="replay" if self.rt.replay_mode else "live",
        )
        board = board.model_copy(
            update={
                "items": ranked_items,
                "decision": probe_out.decision,
                "requests": asks,
                "telemetry": telemetry,
            }
        )
        cb("done", "Coverage verdict and Value-of-Information capture requests ready.", {"verdict": probe_out.decision.verdict})
        return board

    def run_followup(
        self,
        board: LedgerBoard,
        followup_video_path: str | Path | None = None,
        request_id: str | None = None,
        spoken_or_text_answer: str | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> LedgerBoard:
        """Incrementally process a follow-up clip (e.g. 5s laptop underside label) and update board & verdict."""
        cb = progress_cb or _noop_progress
        t0 = time.perf_counter()
        sdir = self.work_dir / board.session_id / f"followup_v{board.board_version + 1}"
        sdir.mkdir(parents=True, exist_ok=True)

        target_req = next(
            (r for r in board.requests if r.request_id == request_id),
            board.requests[0] if board.requests else None,
        )
        if target_req is None:
            return board

        target_item = board.items.get(target_req.item_id) if target_req.item_id else None

        if followup_video_path is not None:
            cb("followup_ingest", "Extracting sharp close-up frames from follow-up clip…", {})
            frames, raw_cnt = extract_frames(
                followup_video_path,
                sdir,
                video_id=f"{board.session_id}_fup",
                sample_fps=3.0,
                max_frames=12,
            )
            cb("followup_extract", f"Gemini 4 Hydrogen reading label on {target_req.item_name or 'item'}…", {})
            obs_list = self.hydrogen.extract_followup_clip(target_req, frames, target_item)
        else:
            frames, raw_cnt = [], 0
            obs_list = []

        if target_item is not None and (obs_list or spoken_or_text_answer):
            best_obs = obs_list[0] if obs_list else None
            best_fr = (
                next((f for f in frames if f.frame_id == best_obs.frame_id), frames[0])
                if (best_obs and frames)
                else None
            )
            brand = (best_obs.brand if best_obs and best_obs.brand else target_item.brand)
            model_txt = (best_obs.model_text if best_obs and best_obs.model_text else target_item.model)
            desc = (
                f"{target_item.description}; {spoken_or_text_answer}"
                if spoken_or_text_answer
                else (best_obs.description if best_obs else target_item.description)
            )
            queries: list[str] = []
            if brand and model_txt:
                queries.append(f"{brand} {model_txt}")
            if spoken_or_text_answer:
                queries.append(f"{brand or ''} {target_item.category} {spoken_or_text_answer}".strip())
            if best_obs:
                queries.extend(best_obs.catalog_queries)
            queries.append(f"{brand or ''} {target_item.category}".strip())

            synth_track = ResolvedItemTrack(
                item_id=target_item.item_id,
                room=target_item.room,
                category=target_item.category,
                observations=obs_list,
                display_name=(
                    f"{brand} {target_item.category.title()} ({model_txt})"
                    if (brand and model_txt)
                    else target_item.display_name
                ),
                brand=brand,
                model_text=model_txt,
                description=desc,
                quantity=target_item.quantity,
                simultaneous_counts=[target_item.quantity],
                best_frame=best_fr,
                best_obs=best_obs,
                catalog_queries=queries,
                narration_conflict=target_item.conflict_detail,
                flags=[f for f in target_item.flags if f not in {"frame_too_blurry", "frame_too_dark"}],
            )

            cb("followup_verify", "Running catalog_exact_match & deterministic verifiers on follow-up evidence…", {})
            f_notes, f_listings, f_patch = self._price_and_verify_track(
                synth_track,
                sdir,
                board.board_version,
                note_seq_start=len(board.notes) + 1,
            )
            board = append_notes_and_listings(board, f_notes, f_listings)
            board = apply_patch(
                board,
                BoardPatch(requested_under=board.board_version, item_patches=[f_patch]),
            )

        rebuilt = rebuild_board(
            board.notes, board.listings_index, board.policy, session_id=board.session_id
        )
        drift = compute_board_drift(board, rebuilt)

        cb("followup_decide", "Re-evaluating Decision-1 coverage verdict & remaining VoI asks…", {})
        probe_out = self.decision.decide_and_probe(board, max_probe_items=self.max_probe_items)
        ranked_items, new_asks = rank_capture_requests(board, probe_out.influences, max_requests=3)

        elapsed = round(board.telemetry.elapsed_seconds + (time.perf_counter() - t0), 2)
        telemetry = board.telemetry.model_copy(
            update={
                "carbon_calls": self.rt.carbon_calls,
                "hydrogen_calls": self.rt.hydrogen_calls,
                "decision_calls": self.rt.decision_calls,
                "shop_mcp_calls": self.shop.call_count,
                "frames_extracted": board.telemetry.frames_extracted + raw_cnt,
                "frames_kept_after_dedup": board.telemetry.frames_kept_after_dedup + len(frames),
                "board_drift_count": drift,
                "elapsed_seconds": elapsed,
            }
        )
        return board.model_copy(
            update={
                "items": ranked_items,
                "decision": probe_out.decision,
                "requests": new_asks,
                "telemetry": telemetry,
            }
        )
