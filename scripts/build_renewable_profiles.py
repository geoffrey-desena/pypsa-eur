# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build per-region capacity factor time series, installable potential and average grid distance for one renewable technology.

The script works at two resolutions: the cutout grid cells of the weather data
and the coarser network regions. For each grid cell it computes the average
capacity factor with [atlite](https://atlite.readthedocs.io/en/latest/) and
takes the eligible land share from [determine_availability_matrix][]. Within
each region the cells are split into a configurable number of resource classes
by capacity factor, and the capacity layout is assumed proportional to eligible
area and capacity factor, since more capacity is expected where the resource is
better. This layout weights the hourly cell time series into one profile per
region and class; the installable potential sums the eligible area times the
configured capacity density; and the average distance of the layout-weighted
cells is measured to the region's representative point for onshore
technologies and to the shoreline for offshore wind. If several turbine or
panel models are configured by year, one profile per model year is produced.
Regions with too little potential or too low a capacity factor are dropped.

| Field | Dimensions | Description |
| --- | --- | --- |
| profile | year, bus, bin, time | Hourly availability factor per unit of installed capacity |
| p_nom_max | bus, bin | Maximal installable capacity in MW |
| average_distance | bus, bin | Average distance in km of the units to the grid bus (onshore) or the shoreline (offshore) |

![](../img/onwind-gridcell.png)

![](../img/offwindac-gridcell.png)

![](../img/offwinddc-gridcell.png)

![](../img/solar-gridcell.png)

![](../img/profile_ts.png)

![](../img/p_nom_max_hist.png)

![](../img/distance_hist.png)
"""

import logging
import time
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from atlite.gis import ExclusionContainer

from scripts._helpers import (
    _simplify_polys,
    configure_logging,
    get_snapshots,
    load_cutout,
    set_scenario_config,
    setup_dask,
)

logger = logging.getLogger(__name__)


def offshore_distance_regions(regions, offshore_buses, network_path):
    """Return distance-reference polygons, including auditable offshore fallbacks.

    Added 2026-11-06 by Codex on behalf of Geoffrey DeSena to unblock Nordic
    fixed-capacity DCOPF profile preparation. Missing offshore IDs reference
    the nearest original same-country onshore polygon, using bus coordinates
    in EPSG:3035. Only this in-memory distance lookup changes; shared land
    allocations, offshore resource regions and electrical connections do not.
    Retain the upstream polygon simplification. Report both distances because
    simplification can remove nearby islands. Above 50 km warns but continues,
    as requested; ambiguous or unavailable assignments remain errors.
    """
    original_crs = regions.crs
    missing = pd.Index(offshore_buses).difference(regions.index)
    assignments = []
    if len(missing):
        with xr.open_dataset(network_path) as network:
            buses = pd.DataFrame(
                {key: network[f"buses_{key}"].to_pandas()
                 for key in ("country", "x", "y")}
            )
        required = regions.index.union(missing)
        if not required.isin(buses.index).all():
            raise ValueError("Distance-reference bus IDs are missing from the network")
        if buses.loc[required, "country"].isna().any() or (
            buses.loc[required, "country"] == ""
        ).any():
            raise ValueError("Distance-reference buses require country labels")
        if not np.isfinite(buses.loc[missing, ["x", "y"]].to_numpy()).all():
            raise ValueError("Offshore fallback buses require finite coordinates")
        projected = regions.to_crs(3035)
        # Search original polygons only; earlier fallback aliases are not candidates.
        original = regions.copy()
        regions = regions.copy()
        for bus in missing:
            country = buses.at[bus, "country"]
            candidates = projected.loc[buses.loc[projected.index, "country"] == country]
            if candidates.empty:
                raise ValueError(f"No onshore distance-reference polygon for {bus} ({country})")
            point = gpd.GeoSeries(
                gpd.points_from_xy([buses.at[bus, "x"]], [buses.at[bus, "y"]]),
                crs=4326,
            ).to_crs(3035).iloc[0]
            distances = candidates.distance(point)
            if not np.isfinite(distances.to_numpy()).all():
                raise ValueError(f"Invalid candidate distance-reference geometry for {bus}")
            nearest = distances.index[
                np.isclose(distances, distances.min(), rtol=0, atol=1e-6)
            ]
            if len(nearest) != 1:
                raise ValueError(f"Ambiguous nearest distance-reference polygon for {bus}: {nearest.tolist()}")
            source = nearest[0]
            regions.loc[bus] = original.loc[source]
            assignments.append((bus, country, source, point, distances.at[source] / 1000))

    regions = regions.loc[offshore_buses].map(
        lambda geometry: _simplify_polys(geometry, minarea=1)
    ).set_crs(original_crs)
    projected = regions.to_crs(3035)
    for bus, country, source, point, original_km in assignments:
        simplified_km = point.distance(projected.loc[bus]) / 1000
        if not np.isfinite(simplified_km):
            raise ValueError(f"Invalid simplified distance-reference geometry for {bus}")
        log = logger.warning if max(original_km, simplified_km) > 50 else logger.info
        log(
            "Offshore distance fallback: bus=%s country=%s reference_polygon=%s "
            "longitude=%.6f latitude=%.6f original_distance_km=%.3f "
            "simplified_distance_km=%.3f exceeds_50_km=%s; assignment retained",
            bus, country, source, buses.at[bus, "x"], buses.at[bus, "y"],
            original_km, simplified_km, max(original_km, simplified_km) > 50,
        )
    return regions


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_renewable_profiles", technology="offwind-ac")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    nprocesses = int(snakemake.threads)
    noprogress = snakemake.config["run"].get("disable_progressbar", True)
    noprogress = noprogress or not snakemake.config["atlite"]["show_progress"]
    technology = snakemake.wildcards.technology
    params = snakemake.params.renewable[technology]
    resource = params["resource"]  # pv panel params / wind turbine params
    resource["show_progress"] = not noprogress

    tech = next(t for t in ["panel", "turbine"] if t in resource)
    models = resource[tech]
    if not isinstance(models, dict):
        models = {0: models}
    resource[tech] = models[next(iter(models))]

    correction_factor = params.get("correction_factor", 1.0)
    capacity_per_sqkm = params["capacity_per_sqkm"]

    if correction_factor != 1.0:
        logger.info(f"correction_factor is set as {correction_factor}")

    dask_kwargs = setup_dask(nprocesses)

    sns = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)

    cutout = load_cutout(snakemake.input.cutout, time=sns)

    availability = xr.open_dataarray(snakemake.input.availability_matrix)

    regions = gpd.read_file(snakemake.input.distance_regions)
    # do not pull up, set_index does not work if geo dataframe is empty
    regions = regions.set_index("name").rename_axis("bus")
    if snakemake.wildcards.technology.startswith("offwind"):
        # for offshore regions, the shortest distance to the shoreline is used
        offshore_regions = availability.coords["bus"].values
        regions = offshore_distance_regions(
            regions.geometry, offshore_regions, snakemake.input.network
        )
    else:
        # for onshore regions, the representative point of the region is used
        regions = regions.representative_point()
    regions = regions.geometry.to_crs(3035)
    buses = regions.index

    area = cutout.grid.to_crs(3035).area / 1e6
    area = xr.DataArray(
        area.values.reshape(cutout.shape), [cutout.coords["y"], cutout.coords["x"]]
    )

    func = getattr(cutout, resource.pop("method"))

    logger.info(
        f"Calculate average capacity factor per grid cell for technology {technology}..."
    )
    start = time.time()

    capacity_factor = correction_factor * func(
        capacity_factor=True,
        dask_kwargs=dask_kwargs,
        **resource,
    )

    duration = time.time() - start
    logger.info(
        f"Completed average capacity factor calculation per grid cell for technology {technology} ({duration:2.2f}s)"
    )

    nbins = params.get("resource_classes", 1)
    logger.info(
        f"Create masks for {nbins} resource classes for technology {technology}..."
    )
    start = time.time()

    fn = snakemake.input.resource_regions
    resource_regions = gpd.read_file(fn).set_index("name").rename_axis("bus").geometry

    # indicator matrix for which cells touch which regions
    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
    I = cutout.availabilitymatrix(resource_regions, ExclusionContainer(), **kwargs)
    I = np.ceil(I)
    cf_by_bus = capacity_factor * I.where(I > 0)

    epsilon = 1e-3
    cf_min, cf_max = (
        cf_by_bus.min(dim=["x", "y"]) - epsilon,
        cf_by_bus.max(dim=["x", "y"]) + epsilon,
    )
    normed_bins = xr.DataArray(np.linspace(0, 1, nbins + 1), dims=["bin"])
    bins = cf_min + (cf_max - cf_min) * normed_bins

    cf_by_bus_bin = cf_by_bus.expand_dims(bin=range(nbins))
    lower_edges = bins[:, :-1]
    upper_edges = bins[:, 1:]
    class_masks = (cf_by_bus_bin >= lower_edges) & (cf_by_bus_bin < upper_edges)

    if nbins == 1:
        bus_bin_mi = pd.MultiIndex.from_product(
            [resource_regions.index, [0]], names=["bus", "bin"]
        )
        class_regions = resource_regions.set_axis(bus_bin_mi)
    else:
        grid = cutout.grid.set_index(["y", "x"])
        class_regions = {}
        for bus, bin_id in product(buses, range(nbins)):
            bus_bin_mask = (
                class_masks.sel(bus=bus, bin=bin_id)
                .stack(spatial=["y", "x"])
                .to_pandas()
            )
            grid_cells = grid.loc[bus_bin_mask]
            geometry = (
                grid_cells.intersection(resource_regions.loc[bus]).union_all().buffer(0)
            )
            class_regions[(bus, bin_id)] = geometry
        class_regions = gpd.GeoSeries(class_regions, crs=4326)
        class_regions.index.names = ["bus", "bin"]
    class_regions.to_file(snakemake.output.class_regions)

    duration = time.time() - start
    logger.info(
        f"Completed resource class calculation for technology {technology} ({duration:2.2f}s)"
    )

    layout = capacity_factor * area * capacity_per_sqkm

    profiles = []
    for year, model in models.items():
        logger.info(
            f"Calculate weighted capacity factor time series for model {model} for technology {technology}..."
        )
        start = time.time()

        resource[tech] = model

        matrix = (availability * class_masks).stack(
            bus_bin=["bus", "bin"], spatial=["y", "x"]
        )

        profile = func(
            matrix=matrix,
            layout=layout,
            index=matrix.indexes["bus_bin"],
            per_unit=True,
            return_capacity=False,
            dask_kwargs=dask_kwargs,
            **resource,
        )
        profile = profile.unstack("bus_bin")

        dim = {"year": [year]}
        profile = profile.expand_dims(dim)

        profiles.append(profile.rename("profile"))

        duration = time.time() - start
        logger.info(
            f"Completed weighted capacity factor time series calculation for model {model} for technology {technology} ({duration:2.2f}s)"
        )

    profiles = xr.merge(profiles)

    logger.info(f"Calculating maximal capacity per bus for technology {technology}")
    p_nom_max = capacity_per_sqkm * availability * class_masks @ area

    logger.info(f"Calculate average distances for technology {technology}.")
    layoutmatrix = (layout * availability * class_masks).stack(
        bus_bin=["bus", "bin"], spatial=["y", "x"]
    )

    coords = cutout.grid.representative_point().to_crs(3035)

    average_distance = []
    bus_bins = layoutmatrix.indexes["bus_bin"]
    for bus, bin in bus_bins:
        row = layoutmatrix.sel(bus=bus, bin=bin).data
        nz_b = row != 0
        row = row[nz_b]
        co = coords[nz_b]
        distances = co.distance(regions[bus]).div(1e3)  # km
        average_distance.append((distances * (row / row.sum())).sum())

    average_distance = xr.DataArray(average_distance, [bus_bins]).unstack("bus_bin")

    ds = xr.merge(
        [
            correction_factor * profiles,
            p_nom_max.rename("p_nom_max"),
            average_distance.rename("average_distance"),
        ]
    )
    # select only buses with some capacity and minimal capacity factor
    mean_profile = ds["profile"].mean("time").max(["year", "bin"])
    sum_potential = ds["p_nom_max"].sum("bin")

    ds = ds.sel(
        bus=(
            (mean_profile > params.get("min_p_max_pu", 0.0))
            & (sum_potential > params.get("min_p_nom_max", 0.0))
        )
    )

    if "clip_p_max_pu" in params:
        min_p_max_pu = params["clip_p_max_pu"]
        ds["profile"] = ds["profile"].where(ds["profile"] >= min_p_max_pu, 0)

    ds.to_netcdf(snakemake.output.profile)
