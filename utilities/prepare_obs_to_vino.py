#!/usr/bin/env python3
"""VINO is the verification in-situ Observation database for operational verification of forecasts.

output should be ready to be injected into vino_import: https://confluence.ecmwf.int/spaces/VER/pages/76406338/vino_import

/home/moz/bin/vino_import --parameter dis --period 3600 * {period_hours} --date {valid_date:%Y%m%d} --time {valid_hour:%02d}00 --dataset {name} {outname}.txt

{outname}.txt is an ASCII file, one station per line, one file per timestamp, with the following format (no header):
<ID>,<Lat>,<Lon>,<Elevation|na>,<Value>
<ID>,<Lat>,<Lon>,<Elevation|na>,<Value>
<ID>,<Lat>,<Lon>,<Elevation|na>,<Value>

It's the database that `quaver` will fetch data from when using observations as reference for score computation.

This script reads a pre-arranged file with observation data from CEMS-Hydro for a given, pre-selected, set of
stations and period of time.

"""
import xarray as xr
import pandas as pd
from dask import delayed, compute


@delayed
def export_df(dataframe: pd.DataFrame, outname: str, fmt: dict[str, str]):
    out = dataframe.to_string(
        col_space=0,
        na_rep="na",
        formatters=fmt,
        index=False,
        header=False,
    )
    with open(f"{outname}", "w") as f:
        f.write(out)


ROOT_DIR="/ec/res4/hpcperm/mojp/obs/efas"
OUTPUT_ROOT = f"{ROOT_DIR}/vino_import"
STATIONS_FILE = "hsix_calib_stations.csv"
OBS_FILES =[
    {
        "source": "Qobs_06.nc",
        "period": 6,
        "variable": "discharge",
        "index_name": "station"
    },
    {
        "source": "Qobs_24.nc",
        "period": 24,
        "variable": "discharge",
        "index_name": "station"
    },
]
INDEX_COL="ObsID"
DATASETS_COORDS = [
    {"name": "test", "lat": "LisfloodY", "lon": "LisfloodX"},
    # {"name": "efas_v5", "lat": "LisfloodY", "lon": "LisfloodX"},
    # {"name": "efas_v4", "lat": "LisfloodY5k", "lon": "LisfloodX5k"},
]
FORMATTERS= {
    INDEX_COL: lambda x: f"{x:d}",
    "lat": lambda x: f"{x:.6f}",
    "lon": lambda x: f"{x:.6f}",
    "value": lambda x: f"{x:.3f}"
}
STEPS=range(32)
VINO_COLS = [INDEX_COL, "lat", "lon", "elevation", "value"]
VINO_CMD="/home/moz/bin/vino_import --parameter dis --table observation --period {period_hours} --date {valid_date:%Y%m%d} --time {valid_hour:02d} --dataset {name} {outname}"


stations = pd.read_csv(f"{ROOT_DIR}/{STATIONS_FILE}", sep=",", header=0, index_col=INDEX_COL)


for vino_dataset in DATASETS_COORDS:
    name = vino_dataset["name"]
    lat_name = vino_dataset["lat"]
    lon_name = vino_dataset["lon"]

    cmds = []
    for obs_file in OBS_FILES:
        source = obs_file["source"]
        period = obs_file["period"]
        period_hours = obs_file["period"] * 3600

        ds = xr.open_dataset(f"{ROOT_DIR}/{source}")
        station_dim = [d for d in ds.dims if d not in ["time"]][0]
        ds = ds.rename({station_dim: INDEX_COL})
        coords_ds = xr.Dataset.from_dataframe(stations.loc[:, [lat_name, lon_name]])
        coords_ds = coords_ds.rename({lat_name: "lat", lon_name: "lon"})
        ds = ds.assign_coords(coords_ds).rename({obs_file["variable"]: "value"})

        delayed = []
        for step in STEPS:
            ds_step = ds.isel(time=step).to_dataframe().dropna()
            valid_date = pd.to_datetime(ds_step.time.values[0])
            valid_hour = valid_date.hour
            ds_step = ds_step.drop(columns=["time"]).reset_index().reindex(columns=VINO_COLS)
            outname = f"{OUTPUT_ROOT}/{name}_{period}h_{valid_date:%Y%m%d%H}.txt"
            print(f"Exporting {len(ds_step)} obs to {outname} for {name} at {valid_date:%Y-%m-%d %H:%M}")
            cmds.append(
                VINO_CMD.format(
                    period_hours=period_hours,
                    valid_date=valid_date,
                    valid_hour=valid_hour,
                    name=name,
                    outname=outname
                )
            )
            delayed.append(
                export_df(ds_step, outname, FORMATTERS)
            )

        compute(*delayed)
    with open(f"{OUTPUT_ROOT}/{name}_cmds.txt", "w") as f:
        for cmd in cmds:
            f.write(cmd + "\n")

#$ parallel -j 4 < {OUTPUT_ROOT}/{name}_cmds.txt