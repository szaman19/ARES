"""Offline tests for ``ares.ext.viz`` run-bundle export and HTML visualizer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ares import Done, Spawn, StateMachine, Task, TaskContext, TaskSpec
from ares.ext.viz import export_run_bundle, main as viz_main, render_visualizer_html, write_visualizer_html


def _run_sample_machine(run_dir: Path) -> None:
    machine = StateMachine()

    @machine.state("root")
    def root(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Spawn:
        _ = task, deps, ctx
        return Spawn(
            tasks=[
                TaskSpec(id="worker_a", state="worker", input={"x": 2}),
                TaskSpec(id="worker_b", state="worker", input={"x": 3}),
            ],
            then=TaskSpec(id="join", state="join"),
        )

    @machine.state("worker", deterministic=True)
    def worker(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        _ = deps
        ctx.record_usage(prompt_tokens=25, completion_tokens=10, llm_calls=1)
        payload = f"DATA:{task.id}:" + ("Z" * 120)
        ref = ctx.store.put_text(payload, name=f"{task.id}.txt")
        return Done(output={"val": task.input["x"] * 10, "blob": payload, "ref": ref.ref_id})

    @machine.state("join", deterministic=True)
    def join(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        _ = task, ctx
        return Done(output={"count": len(deps)})

    res = machine.run(
        TaskSpec(id="root", state="root"),
        run_dir=run_dir,
        max_concurrent=2,
    )
    assert res.status == "done"


def test_export_run_bundle_and_render_standalone_html(tmp_path: Path) -> None:
    run_dir = tmp_path / "sample_run"
    _run_sample_machine(run_dir)

    bundle = export_run_bundle(run_dir, title="Sample Viz Run")
    assert bundle["schema"] == "ares_viz_bundle_v1"
    assert bundle["status"] == "done"
    assert bundle["root_task_id"] == "root"
    assert bundle["usage"]["total_tokens"] == 70
    assert bundle["usage"]["llm_calls"] == 2
    assert set(bundle["tasks"].keys()) == {"root", "worker_a", "worker_b", "join"}
    assert len(bundle["events"]) >= 4
    assert len(bundle["artifacts"]) >= 2

    html_text = render_visualizer_html(real_run_bundle=bundle)
    assert "<!DOCTYPE html>" in html_text
    assert "Overview & Vision" in html_text
    assert "Propose" in html_text
    assert "ChARGe" in html_text
    assert "Story 1: Physics Simulation" in html_text
    assert "Story 2: Literature + Data" in html_text
    assert "Sample Viz Run" in html_text

    # Must be 100% self-contained with zero external network script/stylesheet dependencies.
    assert not re.search(r"<(script|link)[^>]+https?://", html_text, re.IGNORECASE)


def test_write_visualizer_html_and_cli_entrypoint(
    tmp_path: Path,
    capsys: Any,
) -> None:
    run_dir = tmp_path / "cli_run"
    _run_sample_machine(run_dir)

    out_html = tmp_path / "out" / "viz.html"
    written = write_visualizer_html(out_html, run_dir=run_dir)
    assert written == out_html
    assert out_html.exists()
    text = out_html.read_text(encoding="utf-8")
    assert "const ARES_PAYLOAD =" in text

    cli_html = tmp_path / "cli_viz.html"
    bundle_json = tmp_path / "bundle.json"
    rc = viz_main(
        [
            str(run_dir),
            "--output",
            str(cli_html),
            "--bundle-json",
            str(bundle_json),
            "--no-curated",
        ]
    )
    assert rc == 0
    assert "Wrote interactive ARES visualizer" in capsys.readouterr().out
    assert cli_html.exists()
    assert bundle_json.exists()

    parsed_bundle = json.loads(bundle_json.read_text(encoding="utf-8"))
    assert parsed_bundle["schema"] == "ares_viz_bundle_v1"
    assert "worker_a" in parsed_bundle["tasks"]

    cli_html_text = cli_html.read_text(encoding="utf-8")
    assert '"curatedStories": []' in cli_html_text
    assert '"realRunBundle": {' in cli_html_text
