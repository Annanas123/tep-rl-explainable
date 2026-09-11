from __future__ import annotations

"""Create thesis-ready APG-input/RL-reinforcement overlay figures."""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tep_rl.future_scenarios import DEFAULT_NUTS2_BOUNDARIES, DEFAULT_SCENARIO_CATALOG
from tep_rl.line_metadata import parse_linestring


SCENARIO_BY_YEAR = {
    2030: "apg_tyndp_nt_2030",
    2035: "apg_tyndp_nt_2035_midpoint",
    2040: "apg_tyndp_nt_2040",
}
AGENT_LABELS = {"ppo": "PPO", "moppo": "MO-PPO"}
FH_BLUE = "#00649C"
FH_GREEN = "#8BB31D"
FH_GREY = "#72777A"
FH_AMBER = "#FFBF00"
AGENT_COLOURS = {"ppo": FH_BLUE, "moppo": FH_GREEN}
FH_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "fh_sequential", ["#F4F6F7", "#C9DD8D", FH_GREEN, "#2A829F", FH_BLUE]
)
TO_AUSTRIA_LAMBERT = Transformer.from_crs(4326, 31287, always_xy=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--scenario-catalog", type=Path, default=DEFAULT_SCENARIO_CATALOG)
    parser.add_argument("--nuts2-boundaries", type=Path, default=DEFAULT_NUTS2_BOUNDARIES)
    parser.add_argument("--env-mode", choices=("full", "proxy"), default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-seed-frequency", type=float, default=0.4)
    return parser.parse_args()


def _line_coordinates(network: pypsa.Network, line: str) -> list[tuple[float, float]]:
    if line not in network.lines.index:
        return []
    coordinates = parse_linestring(network.lines.at[line, "geometry"])
    if len(coordinates) >= 2:
        return [TO_AUSTRIA_LAMBERT.transform(x, y) for x, y in coordinates]
    bus0 = str(network.lines.at[line, "bus0"])
    bus1 = str(network.lines.at[line, "bus1"])
    fallback = [
        (float(network.buses.at[bus0, "x"]), float(network.buses.at[bus0, "y"])),
        (float(network.buses.at[bus1, "x"]), float(network.buses.at[bus1, "y"])),
    ]
    return (
        [TO_AUSTRIA_LAMBERT.transform(x, y) for x, y in fallback]
        if np.isfinite(np.asarray(fallback, dtype=float)).all()
        else []
    )


def _network_segments(network: pypsa.Network) -> list[list[tuple[float, float]]]:
    return [coords for line in network.lines.index if len(coords := _line_coordinates(network, str(line))) >= 2]


def _dc_upgrades(results_dir: Path, scenario_id: str) -> pd.DataFrame:
    path = results_dir / "dc" / scenario_id / "dc_tep_upgrades.csv"
    if not path.exists():
        return pd.DataFrame(columns=["line", "upgrade_mw"])
    frame = pd.read_csv(path)
    return frame.loc[frame["upgrade_mw"].astype(float) > 1e-6, ["line", "upgrade_mw"]].copy()


def _selected_rl_lines(
    frame: pd.DataFrame,
    top_k: int,
    minimum_seed_frequency: float,
) -> pd.DataFrame:
    positive = frame.loc[frame["mean_upgrade_mw"].astype(float) > 1e-6].copy()
    if positive.empty:
        return positive
    consensus = positive.loc[positive["selection_frequency"] >= minimum_seed_frequency]
    source = consensus if not consensus.empty else positive
    return source.sort_values(["mean_upgrade_mw", "selection_frequency"], ascending=False).head(top_k)


def _scenario_targets(catalog: dict, year: int) -> dict[str, dict[str, float]]:
    scenario = catalog["scenarios"][SCENARIO_BY_YEAR[year]]
    return {
        metric: {str(region): float(value) for region, value in values.items()}
        for metric, values in scenario["regional_targets_mw"].items()
    }


def _plot_overlays(
    results_dir: Path,
    network: pypsa.Network,
    nuts2: gpd.GeoDataFrame,
    catalog: dict,
    lines: pd.DataFrame,
    policies: pd.DataFrame,
    env_mode: str,
    top_k: int,
    minimum_seed_frequency: float,
    output_dir: Path,
) -> None:
    years = [2030, 2035, 2040]
    agents = [agent for agent in ("ppo", "moppo") if agent in set(lines["agent"])]
    if not agents:
        raise ValueError(f"No PPO or MO-PPO line summaries are available for {env_mode}.")
    targets = {year: _scenario_targets(catalog, year) for year in years}
    renewable = {
        year: {
            region: (targets[year]["onwind"][region] + targets[year]["solar"][region]) / 1000.0
            for region in targets[year]["onwind"]
        }
        for year in years
    }
    peak = {
        year: {region: value / 1000.0 for region, value in targets[year]["peak_load"].items()}
        for year in years
    }
    maximum_renewable = max(value for values in renewable.values() for value in values.values())
    maximum_peak = max(value for values in peak.values() for value in values.values())
    normalise = Normalize(vmin=0.0, vmax=maximum_renewable)
    cmap = FH_SEQUENTIAL
    existing_segments = _network_segments(network)
    mode_lines = lines.loc[lines["env_mode"].eq(env_mode)]
    global_rl_max = max(float(mode_lines["mean_upgrade_mw"].max()), 1.0)
    dc_frames = {year: _dc_upgrades(results_dir, SCENARIO_BY_YEAR[year]) for year in years}
    global_dc_max = max(
        [float(frame["upgrade_mw"].max()) for frame in dc_frames.values() if not frame.empty] or [1.0]
    )

    fig, axes = plt.subplots(
        len(agents),
        len(years),
        figsize=(11.7, 3.85 * len(agents)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, agent in enumerate(agents):
        for column, year in enumerate(years):
            ax = axes[row, column]
            frame = nuts2.copy()
            frame["renewable_gw"] = frame["NUTS_ID"].map(renewable[year]).astype(float)
            frame["peak_gw"] = frame["NUTS_ID"].map(peak[year]).astype(float)
            frame.plot(
                ax=ax,
                column="renewable_gw",
                cmap=cmap,
                norm=normalise,
                edgecolor=FH_GREY,
                linewidth=0.45,
                zorder=1,
            )
            ax.add_collection(
                LineCollection(existing_segments, colors="#858585", linewidths=0.25, alpha=0.30, zorder=2)
            )
            bubble_sizes = 8.0 + 48.0 * frame["peak_gw"].to_numpy(dtype=float) / maximum_peak
            ax.scatter(
                frame["centroid_x"],
                frame["centroid_y"],
                s=bubble_sizes,
                facecolors="none",
                edgecolors="#CC79A7",
                linewidths=0.8,
                zorder=3,
            )

            subset = lines.loc[
                lines["scenario_year"].eq(year)
                & lines["env_mode"].eq(env_mode)
                & lines["agent"].eq(agent)
            ]
            selected = _selected_rl_lines(subset, top_k, minimum_seed_frequency)
            for _, record in selected.iterrows():
                coordinates = _line_coordinates(network, str(record["line"]))
                if len(coordinates) < 2:
                    continue
                frequency = float(record["selection_frequency"])
                width = 1.2 + 4.0 * float(record["mean_upgrade_mw"]) / global_rl_max
                ax.plot(
                    *zip(*coordinates),
                    color=AGENT_COLOURS[agent],
                    linewidth=width,
                    alpha=0.30 + 0.70 * frequency,
                    solid_capstyle="round",
                    zorder=5,
                )

            for _, record in dc_frames[year].sort_values("upgrade_mw", ascending=False).head(top_k).iterrows():
                coordinates = _line_coordinates(network, str(record["line"]))
                if len(coordinates) < 2:
                    continue
                width = 1.0 + 3.0 * float(record["upgrade_mw"]) / global_dc_max
                ax.plot(
                    *zip(*coordinates),
                    color=FH_AMBER,
                    linewidth=width,
                    linestyle=(0, (4, 2)),
                    alpha=0.90,
                    zorder=6,
                )

            policy_row = policies.loc[
                policies["scenario_year"].eq(year)
                & policies["env_mode"].eq(env_mode)
                & policies["agent"].eq(agent)
            ]
            investment = float(policy_row["total_investment_mean"].iloc[0]) if not policy_row.empty else np.nan
            spread = (
                float(policy_row["total_investment_mean_std_across_seeds"].iloc[0])
                if not policy_row.empty
                else np.nan
            )
            year_label = "2035 interpolated" if year == 2035 else f"{year} APG NT"
            ax.set_title(
                f"{AGENT_LABELS[agent]} | {year_label}\nmean reinforcement: {investment:.0f} +/- {spread:.0f} MW",
                fontsize=9.0,
                loc="left",
                fontweight="normal",
            )
            ax.set_aspect("equal")
            ax.set_axis_off()

    scalar_mappable = matplotlib.cm.ScalarMappable(norm=normalise, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.035,
        pad=0.015,
        aspect=42,
    )
    colorbar.set_label("Regional installed onshore wind plus photovoltaic capacity (GW)")
    handles = [
        Line2D([0], [0], color=AGENT_COLOURS[agent], linewidth=3.0, label=f"{AGENT_LABELS[agent]} mean reinforcement")
        for agent in agents
    ]
    if any(not frame.empty for frame in dc_frames.values()):
        handles.append(Line2D([0], [0], color=FH_AMBER, linewidth=2.4, linestyle=(0, (4, 2)), label="Linear reinforcement reference"))
    handles.extend(
        [
            Line2D([0], [0], color="#858585", linewidth=0.8, alpha=0.5, label="Existing transmission corridor"),
            Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#CC79A7", linestyle="none", label="Regional peak load"),
        ]
    )
    fig.legend(handles=handles, frameon=False, ncol=min(5, len(handles)), loc="upper center", bbox_to_anchor=(0.5, 1.035))
    fig.savefig(output_dir / f"future_reinforcement_overlay_{env_mode}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"future_reinforcement_overlay_{env_mode}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_investment_trajectory(
    results_dir: Path,
    policies: pd.DataFrame,
    env_mode: str,
    output_dir: Path,
) -> None:
    frame = policies.loc[policies["env_mode"].eq(env_mode)].copy()
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.3, 3.8))
    maximum_value = 0.0
    styles = {"ppo": ((0, (5, 2)), "o"), "moppo": ((0, (5, 2, 1.5, 2)), "s")}
    for agent in ("ppo", "moppo"):
        subset = frame.loc[frame["agent"].eq(agent)].sort_values("scenario_year")
        if subset.empty:
            continue
        maximum_value = max(maximum_value, float(subset["total_investment_mean"].max()))
        linestyle, marker = styles[agent]
        ax.plot(
            subset["scenario_year"],
            subset["total_investment_mean"],
            marker=marker,
            linewidth=2.0,
            color=AGENT_COLOURS[agent],
            linestyle=linestyle,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=f"{AGENT_LABELS[agent]} five-seed mean",
        )
    dc_years: list[int] = []
    dc_values: list[float] = []
    for year, scenario_id in SCENARIO_BY_YEAR.items():
        summary_path = results_dir / "dc" / scenario_id / "dc_tep_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            dc_years.append(year)
            dc_values.append(float(summary["total_upgrade_mw"]))
    if dc_years:
        ax.plot(dc_years, dc_values, marker="D", linewidth=1.8, linestyle="--", color=FH_GREY, label="Linear reinforcement reference")
        maximum_value = max(maximum_value, max(dc_values))
    ax.set_xticks([2030, 2035, 2040])
    ax.set_xlabel("Target-year input scenario", fontsize=10)
    ax.set_ylabel("Reinforcement (MW)", fontsize=10)
    ax.set_ylim(0.0, max(1.0, maximum_value * 1.18))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"future_reinforcement_trajectory_{env_mode}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"future_reinforcement_trajectory_{env_mode}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    lines_path = results_dir / "line_summary.csv"
    policies_path = results_dir / "policy_summary.csv"
    if not lines_path.exists() or not policies_path.exists():
        raise FileNotFoundError("Run the RL scenario evaluation before creating figures.")
    lines = pd.read_csv(lines_path)
    policies = pd.read_csv(policies_path)
    available_modes = [mode for mode in ("full", "proxy") if mode in set(lines["env_mode"])]
    env_mode = args.env_mode or ("full" if "full" in available_modes else available_modes[0])
    network = pypsa.Network(args.network)
    nuts2 = gpd.read_file(args.nuts2_boundaries)
    nuts2 = nuts2.loc[nuts2["CNTR_CODE"].eq("AT") & nuts2["LEVL_CODE"].eq(2)].copy().to_crs(epsg=31287)
    centroids = nuts2.geometry.centroid
    nuts2["centroid_x"] = centroids.x
    nuts2["centroid_y"] = centroids.y
    catalog = json.loads(args.scenario_catalog.read_text(encoding="utf-8"))
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_overlays(
        results_dir,
        network,
        nuts2,
        catalog,
        lines,
        policies,
        env_mode,
        args.top_k,
        args.minimum_seed_frequency,
        output_dir,
    )
    _plot_investment_trajectory(results_dir, policies, env_mode, output_dir)
    print(f"Future reinforcement figures written to {output_dir}")


if __name__ == "__main__":
    main()
