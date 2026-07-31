#!/usr/bin/env python3
"""Explore dot products, feature superposition, and high-dimensional capacity."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Patch
from matplotlib.widgets import Button, Slider

QUERY_COLOR = "#0072B2"
ACTIVE_COLOR = "#D55E00"
INACTIVE_COLOR = "#9AA0A6"
GUIDE_COLOR = "#666666"
CAPACITY_COLOR = "#009E73"


@dataclass(frozen=True)
class SuperpositionState:
    """One deterministic random codebook and one sparse activation."""

    directions: np.ndarray
    active_indices: np.ndarray
    activations: np.ndarray
    superposed: np.ndarray
    decoded: np.ndarray
    interference: np.ndarray
    gram: np.ndarray


def random_unit_vectors(
    rng: np.random.Generator,
    count: int,
    dimension: int,
) -> np.ndarray:
    """Return random feature directions with unit Euclidean norm."""

    vectors = rng.normal(size=(count, dimension))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def simulate_superposition(
    dimension: int,
    feature_count: int,
    active_count: int,
    seed: int,
) -> SuperpositionState:
    """Encode sparse features as a vector sum and decode with dot products."""

    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if feature_count < 2:
        raise ValueError("feature_count must be at least 2")
    if not 1 <= active_count <= feature_count:
        raise ValueError("active_count must be between 1 and feature_count")

    rng = np.random.default_rng(seed)
    directions = random_unit_vectors(rng, feature_count, dimension)
    active_indices = np.sort(
        rng.choice(feature_count, size=active_count, replace=False)
    )

    activations = np.zeros(feature_count)
    activations[active_indices] = 1.0

    # Each row of directions is one feature f_i. Encoding and decoding are:
    #     x       = F^T a
    #     scores  = F x = F F^T a
    superposed = directions.T @ activations
    decoded = directions @ superposed
    interference = decoded - activations
    gram = directions @ directions.T

    return SuperpositionState(
        directions=directions,
        active_indices=active_indices,
        activations=activations,
        superposed=superposed,
        decoded=decoded,
        interference=interference,
        gram=gram,
    )


def rough_pairwise_capacity_log2(
    dimension: np.ndarray | float,
    tolerance: float,
) -> np.ndarray | float:
    """Approximate log2 codebook size for bounded pairwise cosine overlap.

    For random unit vectors, cos(theta) is approximately Normal(0, 1/d).
    Applying a union-bound rule of thumb across all vector pairs gives

        N ~= exp((d - 1) * epsilon^2 / 4).

    This is an asymptotic intuition aid, not an exact packing bound.
    """

    return (np.asarray(dimension) - 1.0) * tolerance**2 / (4.0 * math.log(2.0))


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    """Return the upper-triangle entries without the diagonal."""

    return matrix[np.triu_indices(matrix.shape[0], k=1)]


class SuperpositionExplorer:
    """Interactive Matplotlib explanation with linked controls."""

    def __init__(
        self,
        *,
        dimension: int,
        feature_count: int,
        active_count: int,
        theta_degrees: float,
        tolerance: float,
        seed: int,
    ) -> None:
        self.seed = seed
        self.generation = 0
        self.state: SuperpositionState | None = None

        self.figure, axes = plt.subplots(2, 2, figsize=(14, 9))
        (
            (self.dot_axis, self.gram_axis),
            (self.decode_axis, self.capacity_axis),
        ) = axes
        self.figure.subplots_adjust(
            bottom=0.23,
            hspace=0.42,
            left=0.07,
            right=0.97,
            top=0.90,
            wspace=0.25,
        )
        self.figure.suptitle(
            "Feature superposition: dot products recover sparse signals",
            fontsize=16,
            fontweight="bold",
        )

        self.theta_slider = Slider(
            self.figure.add_axes((0.08, 0.145, 0.35, 0.025)),
            "angle θ",
            0,
            180,
            valinit=theta_degrees,
            valstep=1,
        )
        self.dimension_slider = Slider(
            self.figure.add_axes((0.57, 0.145, 0.33, 0.025)),
            "dimensions d",
            8,
            256,
            valinit=dimension,
            valstep=8,
        )
        self.features_slider = Slider(
            self.figure.add_axes((0.08, 0.095, 0.35, 0.025)),
            "feature directions N",
            16,
            512,
            valinit=feature_count,
            valstep=8,
        )
        self.active_slider = Slider(
            self.figure.add_axes((0.57, 0.095, 0.22, 0.025)),
            "active features k",
            1,
            12,
            valinit=active_count,
            valstep=1,
        )
        self.tolerance_slider = Slider(
            self.figure.add_axes((0.08, 0.045, 0.35, 0.025)),
            "allowed |cos θ| ε",
            0.2,
            0.8,
            valinit=tolerance,
            valstep=0.05,
        )
        self.resample_button = Button(
            self.figure.add_axes((0.81, 0.035, 0.12, 0.055)),
            "New directions",
        )

        self.theta_slider.on_changed(self._angle_changed)
        self.dimension_slider.on_changed(self._simulation_changed)
        self.features_slider.on_changed(self._simulation_changed)
        self.active_slider.on_changed(self._simulation_changed)
        self.tolerance_slider.on_changed(self._capacity_changed)
        self.resample_button.on_clicked(self._resample)

        self._draw_dot_product()
        self._regenerate_and_draw()

    @property
    def dimension(self) -> int:
        return round(self.dimension_slider.val)

    @property
    def feature_count(self) -> int:
        return round(self.features_slider.val)

    @property
    def active_count(self) -> int:
        return min(round(self.active_slider.val), self.feature_count)

    @property
    def tolerance(self) -> float:
        return float(self.tolerance_slider.val)

    def _angle_changed(self, _value: float) -> None:
        self._draw_dot_product()
        self.figure.canvas.draw_idle()

    def _simulation_changed(self, _value: float) -> None:
        self._regenerate_and_draw()
        self.figure.canvas.draw_idle()

    def _capacity_changed(self, _value: float) -> None:
        self._draw_capacity()
        self.figure.canvas.draw_idle()

    def _resample(self, _event: object) -> None:
        self.generation += 1
        self._regenerate_and_draw()
        self.figure.canvas.draw_idle()

    def _regenerate_and_draw(self) -> None:
        self.state = simulate_superposition(
            dimension=self.dimension,
            feature_count=self.feature_count,
            active_count=self.active_count,
            seed=self.seed + self.generation,
        )
        self._draw_gram_matrix()
        self._draw_decode_scores()
        self._draw_capacity()

    def _draw_dot_product(self) -> None:
        axis = self.dot_axis
        axis.clear()

        theta_degrees = float(self.theta_slider.val)
        theta = math.radians(theta_degrees)
        cosine = math.cos(theta)
        key = np.array([cosine, math.sin(theta)])

        axis.axhline(0, color="#C6C6C6", linewidth=0.8)
        axis.axvline(0, color="#C6C6C6", linewidth=0.8)
        axis.annotate(
            "",
            xy=(1, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": QUERY_COLOR, "lw": 3},
        )
        axis.annotate(
            "",
            xy=key,
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": ACTIVE_COLOR, "lw": 3},
        )
        axis.plot(
            [key[0], key[0]],
            [0, key[1]],
            linestyle=":",
            color=GUIDE_COLOR,
            linewidth=1.5,
        )
        axis.plot(
            [0, key[0]],
            [0, 0],
            color=ACTIVE_COLOR,
            linewidth=5,
            alpha=0.45,
        )
        axis.add_patch(
            Arc(
                (0, 0),
                0.65,
                0.65,
                theta1=0,
                theta2=theta_degrees,
                color=GUIDE_COLOR,
                linewidth=1.2,
            )
        )

        label_radius = 0.43
        label_angle = theta / 2.0
        axis.text(
            label_radius * math.cos(label_angle),
            label_radius * math.sin(label_angle),
            "θ",
            ha="center",
            va="center",
        )
        axis.text(1.04, -0.04, "query q", color=QUERY_COLOR, va="top")
        axis.text(
            key[0] * 1.07,
            key[1] * 1.07,
            "feature f",
            color=ACTIVE_COLOR,
            ha="center",
        )
        axis.text(
            cosine / 2.0,
            -0.13,
            f"projection = {cosine:+.2f}",
            ha="center",
            va="top",
        )
        axis.text(
            0.02,
            0.97,
            f"q · f = ‖q‖ ‖f‖ cos θ = cos({theta_degrees:.0f}°) = {cosine:+.2f}",
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )

        axis.set(
            aspect="equal",
            xlim=(-1.25, 1.25),
            ylim=(-0.35, 1.25),
            xticks=[],
            yticks=[],
            title="1. A dot product measures directional overlap",
        )
        for spine in axis.spines.values():
            spine.set_visible(False)

    def _draw_gram_matrix(self) -> None:
        assert self.state is not None
        axis = self.gram_axis
        axis.clear()

        active = self.state.active_indices
        inactive = np.setdiff1d(
            np.arange(self.feature_count),
            active,
            assume_unique=True,
        )
        display_count = min(32, self.feature_count)
        display_indices = np.concatenate(
            [active, inactive[: max(0, display_count - len(active))]]
        )[:display_count]
        visible_gram = self.state.gram[np.ix_(display_indices, display_indices)]

        axis.imshow(
            visible_gram,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        boundary = len(active) - 0.5
        axis.axhline(boundary, color="black", linewidth=1.2)
        axis.axvline(boundary, color="black", linewidth=1.2)

        all_off_diagonal = off_diagonal_values(self.state.gram)
        rms = float(np.sqrt(np.mean(all_off_diagonal**2)))
        maximum = float(np.max(np.abs(all_off_diagonal)))
        axis.text(
            0.02,
            0.98,
            f"{self.feature_count} directions in {self.dimension}D  |  "
            f"off-diagonal RMS {rms:.2f}, max {maximum:.2f}",
            transform=axis.transAxes,
            va="top",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

        if display_count <= 20:
            labels = [f"f{index}" for index in display_indices]
            ticks = np.arange(display_count)
            axis.set_xticks(ticks, labels, rotation=90, fontsize=7)
            axis.set_yticks(ticks, labels, fontsize=7)
        else:
            axis.set_xticks([])
            axis.set_yticks([])

        axis.set_title("2. The Gram matrix F Fᵀ contains every pairwise cosine")
        axis.set_xlabel("active features are grouped first; diagonal = 1")

    def _draw_decode_scores(self) -> None:
        assert self.state is not None
        axis = self.decode_axis
        axis.clear()

        indices = np.arange(self.feature_count)
        active_mask = self.state.activations.astype(bool)
        colors = np.where(active_mask, ACTIVE_COLOR, INACTIVE_COLOR)
        axis.bar(indices, self.state.decoded, color=colors, width=0.8)
        axis.axhline(0, color=GUIDE_COLOR, linewidth=0.8)
        axis.axhline(
            0.5,
            color=QUERY_COLOR,
            linestyle="--",
            linewidth=1.5,
            label="example detection threshold",
        )

        inactive_mask = ~active_mask
        false_positives = int(np.count_nonzero(self.state.decoded[inactive_mask] > 0.5))
        misses = int(np.count_nonzero(self.state.decoded[active_mask] <= 0.5))
        cross_talk_rms = float(np.sqrt(np.mean(self.state.interference**2)))

        axis.text(
            0.01,
            0.97,
            "x = Σ active fⱼ\nfᵢ · x = activationᵢ + cross-talk",
            transform=axis.transAxes,
            va="top",
        )
        axis.text(
            0.01,
            0.80,
            f"cross-talk RMS {cross_talk_rms:.2f}  |  "
            f"false positives {false_positives}  |  misses {misses}",
            transform=axis.transAxes,
            va="top",
            color=GUIDE_COLOR,
        )
        axis.legend(
            handles=[
                Patch(color=ACTIVE_COLOR, label="active feature"),
                Patch(color=INACTIVE_COLOR, label="inactive feature"),
                Line2D(
                    [0],
                    [0],
                    color=QUERY_COLOR,
                    linestyle="--",
                    label="threshold 0.5",
                ),
            ],
            loc="upper right",
            frameon=False,
        )

        lower = min(-0.45, float(np.min(self.state.decoded)) - 0.1)
        upper = max(1.65, float(np.max(self.state.decoded)) + 0.35)
        axis.set(
            xlim=(-1, self.feature_count),
            ylim=(lower, upper),
            xlabel="feature index i",
            ylabel="decoded score fᵢ · x",
            title="3. Sparse features survive; other directions add cross-talk",
        )

    def _draw_capacity(self) -> None:
        axis = self.capacity_axis
        axis.clear()

        dimensions = np.arange(4, 513)
        log2_capacity = rough_pairwise_capacity_log2(
            dimensions,
            self.tolerance,
        )
        selected_log2 = float(
            rough_pairwise_capacity_log2(self.dimension, self.tolerance)
        )
        doubling_dimensions = 4.0 * math.log(2.0) / self.tolerance**2

        axis.plot(
            dimensions,
            log2_capacity,
            color=CAPACITY_COLOR,
            linewidth=2.5,
        )
        axis.scatter(
            [self.dimension],
            [selected_log2],
            color=ACTIVE_COLOR,
            s=70,
            zorder=3,
        )
        axis.annotate(
            f"d={self.dimension}: roughly 2^{selected_log2:.1f} directions",
            xy=(self.dimension, selected_log2),
            xytext=(12, 16),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": GUIDE_COLOR},
        )
        axis.text(
            0.02,
            0.97,
            "log₂ N ≈ (d−1) ε² / (4 ln 2)  "
            f"→ capacity doubles about every {doubling_dimensions:.1f} dimensions",
            transform=axis.transAxes,
            va="top",
        )
        axis.text(
            0.98,
            0.05,
            "rough random-vector scaling—not exact capacity\n"
            "and not exponentially many simultaneous dense values",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color=GUIDE_COLOR,
        )

        axis.set(
            xlim=(4, 512),
            ylim=(0, max(5.0, float(log2_capacity[-1]) * 1.08)),
            xlabel="representation dimension d",
            ylabel="feature directions (log₂ scale)",
            title=("4. Fixed |cos θ| tolerance permits an exponential codebook"),
        )
        axis.grid(alpha=0.25)


def run_self_test() -> None:
    """Check the numerical identities used by the visualization."""

    state = simulate_superposition(
        dimension=64,
        feature_count=96,
        active_count=4,
        seed=17,
    )
    np.testing.assert_allclose(
        np.linalg.norm(state.directions, axis=1),
        np.ones(96),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        state.superposed,
        state.directions.T @ state.activations,
    )
    np.testing.assert_allclose(
        state.decoded,
        state.directions @ state.superposed,
    )
    np.testing.assert_allclose(
        state.decoded,
        state.activations + state.interference,
    )

    exact_directions = np.eye(8)
    exact_activations = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=float)
    exact_superposed = exact_directions.T @ exact_activations
    exact_decoded = exact_directions @ exact_superposed
    np.testing.assert_allclose(exact_decoded, exact_activations)

    tolerance = 0.5
    doubling_dimensions = 4.0 * math.log(2.0) / tolerance**2
    before = float(rough_pairwise_capacity_log2(100, tolerance))
    after = float(rough_pairwise_capacity_log2(100 + doubling_dimensions, tolerance))
    np.testing.assert_allclose(after - before, 1.0)
    print("superposition visualization self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize cosine dot products, sparse feature superposition, "
            "cross-talk, and exponential high-dimensional codebooks."
        )
    )
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--features", type=int, default=96)
    parser.add_argument("--active", type=int, default=4)
    parser.add_argument("--theta", type=float, default=55.0)
    parser.add_argument("--tolerance", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--save",
        type=Path,
        help="save the figure to this path instead of opening a window",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="construct the visualization without opening a window",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run numerical checks and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    if args.dimension < 8 or args.dimension > 256:
        raise SystemExit("--dimension must be between 8 and 256")
    if args.features < 16 or args.features > 512:
        raise SystemExit("--features must be between 16 and 512")
    if args.active < 1 or args.active > 12 or args.active > args.features:
        raise SystemExit("--active must be between 1 and min(12, features)")
    if not 0 <= args.theta <= 180:
        raise SystemExit("--theta must be between 0 and 180 degrees")
    if not 0.2 <= args.tolerance <= 0.8:
        raise SystemExit("--tolerance must be between 0.2 and 0.8")

    explorer = SuperpositionExplorer(
        dimension=args.dimension,
        feature_count=args.features,
        active_count=args.active,
        theta_degrees=args.theta,
        tolerance=args.tolerance,
        seed=args.seed,
    )
    # Keep callbacks alive for the full lifetime of the Matplotlib figure.
    explorer.figure._superposition_explorer = explorer  # type: ignore[attr-defined]

    if args.save is not None:
        destination = args.save.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        explorer.figure.savefig(destination, dpi=160, bbox_inches="tight")
        print(f"saved {destination}")

    if args.no_show or args.save is not None:
        plt.close(explorer.figure)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
