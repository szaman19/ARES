"""Domain plug-ins for the ARES Hackathon harness."""

from domain.artifacts import render_candidate_slice_plot
from domain.micro_bench import MicroBenchDomain
from domain.protocol import Domain

__all__ = ["Domain", "MicroBenchDomain", "render_candidate_slice_plot"]
