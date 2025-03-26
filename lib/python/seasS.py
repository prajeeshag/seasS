import calendar
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import requests
import typer
import xarray as xr
from cdo import Cdo
from dateutil.relativedelta import relativedelta

app = typer.Typer()

NMME_MODELS = [
    "CanSIPS-IC4",
    "COLA-RSMAS-CESM1",
    "GFDL-SPEAR",
    "NCEP-CFSv2",
    "NASA-GEOSS2S",
]


PERCENTILES = [33, 66]


def get_url(model: str, year: int, mon_abbr: str, lstart: int, lend: int, fldname: str):
    lead_range = f"L/{lstart}.5/{lend}.5/RANGEEDGES/"
    url1 = f"http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.{model}/"
    area = "Y/5/40/RANGE/X/20/60/RANGEEDGES"
    url2 = f".MONTHLY/.{fldname}/{area}/S/%28%201%20{mon_abbr}%20{year}%29/VALUES/{lead_range}%5BL%5D/average/data.nc"
    if model == "COLA-RSMAS-CESM1":
        return f"{url1}{url2}"
    elif model == "NCEP-CFSv2":
        return f"{url1}.FORECAST/.EARLY_MONTH_SAMPLES/{url2}"
    else:
        return f"{url1}.FORECAST/{url2}"


def download_data(url: str, output_path: str) -> None:
    if os.path.exists(output_path):
        return
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
    Path(output_path).parent.mkdir(
        parents=True, exist_ok=True
    )  # Create directories if they don't exist
    temp_file = f"{output_path}.download"
    with open(temp_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    shutil.move(temp_file, output_path)


def get_hist_pctl_path(
    model: str, month: int, lstart: int, lend: int, root_path: str, fldname: str
):
    return f"{root_path}/{fldname}/hist/{model}/percentiles/{month:02d}/L-{lstart}.5-{lend}.5/{model}-{month:02d}-L-{lstart}.5-{lend}.5_percentiles.nc"


def compute_frequencies(data_file, hist_pctl_data, fldname):
    ds = xr.open_dataset(data_file, decode_times=False)
    hist_ds = xr.open_dataset(hist_pctl_data)
    var = ds[fldname].squeeze()
    if len(var.shape) > 3 or len(var.shape) < 2:
        raise SystemError(f"var.shape {var.shape}")
    ne = 1
    if len(var.shape) == 3:
        ne = var.shape[0]
    res = []
    f33 = None
    f66 = None
    for p in PERCENTILES:
        pname = f"p{p}"
        fname = f"f{p}"
        pvar = hist_ds[pname]
        fvar = var.copy(deep=True)
        if p < 50:
            fvar.values = var.values < pvar.values
        else:
            fvar.values = var.values > pvar.values

        fvar = fvar.sum(dim="M")
        fvar.name = fname
        if fname == "f33":
            f33 = fvar
        if fname == "f66":
            f66 = fvar
        # res.append(fvar)

    fvar = f33.copy(deep=True)
    fvar.values = fix_vectors(f33.values, f66.values)
    fvar.name = "tercile_probability"
    res.append(fvar)

    return ne, res


def fix_vectors(
    f33: np.ndarray,
    f66: np.ndarray,
) -> np.ndarray:
    fnorm = 1.0 - f66 - f33
    stacked = np.stack([f33, fnorm, f66], axis=0)
    max_indices = np.argmax(stacked, axis=0)
    result = np.empty_like(f33, dtype=float)
    result[max_indices == 0] = -f33[max_indices == 0]
    result[max_indices == 1] = np.nan
    result[max_indices == 2] = f66[max_indices == 2]
    return result


def get_fcst_file_path(year, month, lstart, lend, model, root_path, fldname):
    data_file = f"{root_path}/{fldname}/fcst/{model}/averages/{year}/{month:02d}/L-{lstart}.5-{lend}.5/data.nc"
    Path(data_file).parent.mkdir(
        parents=True, exist_ok=True
    )  # Create directories if they don't exist
    return data_file


def get_mme_freq_path(acc: int, data_root: str, fldname: str):
    return f"{data_root}/{fldname}/live/{fldname}-{acc}monthly-frequencies.nc"


@app.command()
def get_fcst_data(year: int, month: int, acc: int, data_root: str, fldname: str):
    mon_abbr = calendar.month_abbr[month]
    for model in NMME_MODELS:
        for lstart in range(1, 8 - acc):
            lend = lstart + acc - 1
            url = get_url(model, year, mon_abbr, lstart, lend, fldname)
            data_file = get_fcst_file_path(
                year, month, lstart, lend, model, data_root, fldname
            )
            print(f"downloading {data_file}")
            download_data(url, data_file)


@app.command()
def process_mme_freq(
    year: int,
    month: int,
    acc: int,
    data_root: str,
    fldname: str,
):
    datasets = []
    for lstart in range(1, 8 - acc):
        lend = lstart + acc - 1
        for n, model in enumerate(NMME_MODELS):
            data_file = get_fcst_file_path(
                year, month, lstart, lend, model, data_root, fldname
            )
            hist_pctl_data = get_hist_pctl_path(
                model, month, lstart, lend, data_root, fldname
            )
            if n == 0:
                ne, fr = compute_frequencies(data_file, hist_pctl_data, fldname)
            else:
                nne, ffr = compute_frequencies(data_file, hist_pctl_data, fldname)
                ne += nne
                for i in range(len(fr)):
                    fr[i].values += ffr[i].values
        for i in range(len(fr)):
            fr[i].values = fr[i].values / float(ne)

        tstart = add_months(year, month, lstart)
        tend = add_months(year, month, lend)
        dataset = xr.Dataset({da.name: da for da in fr})
        dataset = dataset.drop_vars(["S"])
        dataset["tstart"] = (("S"), [tstart])
        dataset["tend"] = (("S"), [tend])
        datasets.append(dataset)

    dataset = xr.concat(datasets, dim="S")
    dataset = dataset.rename({"S": "time", "X": "lon", "Y": "lat"})
    ntime = len(datasets)
    ym = str(dataset["tstart"][0].values)
    dates = pd.date_range(start=f"{ym[0:4]}-{ym[4:6]}-01", periods=ntime, freq="MS")
    ref_time = pd.Timestamp("1960-01-01")
    time_units = (dates - ref_time).days
    dataset.coords["time"] = ("time", time_units)
    dataset.coords["time"].attrs["units"] = "days since 1960-01-01"
    dataset.coords["time"].attrs["calendar"] = "gregorian"
    data_root1 = f"{data_root}/mme/freq/{year}/{month:02d}"
    output = get_mme_freq_path(acc, data_root1, fldname)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output, unlimited_dims=["time"])


def add_months(year: int, month: int, months_to_add: int) -> int:
    new_date = datetime(year, month, 1) + relativedelta(months=months_to_add)
    return new_date.year * 100 + new_date.month


@app.command()
def mme_freq_path(acc: int, data_root: str, fldname: str):
    print(get_mme_freq_path(acc, data_root, fldname))


class C3S:
    MODELS = {
        "dwd": {"system": "21", "time_name": "forecast_reference_time"},
        "meteo_france": {"system": "8", "time_name": "forecast_reference_time"},
        "cmcc": {"system": "35", "time_name": "forecast_reference_time"},
        "ukmo": {"system": "603", "time_name": "indexing_time"},
        "ecmwf": {"system": "51", "time_name": "forecast_reference_time"},
    }
    PREFIX = "C3S"
    AREA = [40, 10, 0, 70]

    def download_hist(
        self,
        start_year: int,
        end_year: int,
        data_root: str,
        grid_file: str,
    ):
        client = cdsapi.Client()
        cdo = Cdo()
        for model, info in self.MODELS.items():
            for month in map(str, list(range(1, 13))):
                for leadtime_month in map(str, list(range(2, 7))):
                    system = info["system"]
                    time_name = info["time_name"]
                    target = (
                        Path(data_root)
                        / self.PREFIX
                        / "hist"
                        / f"{model}_{month}_{leadtime_month}.nc"
                    )
                    if os.path.exists(target):
                        continue
                    dataset = "seasonal-monthly-single-levels"
                    request = {
                        "originating_centre": model,
                        "system": system,
                        "variable": ["total_precipitation"],
                        "product_type": ["monthly_mean"],
                        "year": list(map(str, list(range(start_year, end_year + 1)))),
                        "month": [month],
                        "leadtime_month": [leadtime_month],
                        "data_format": "netcdf",
                        "area": [40, 10, 0, 70],
                    }
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
                    temp_file = f"{target}.download"
                    temp_file1 = f"{target}.1"
                    temp_file2 = f"{target}.2"
                    temp_file3 = f"{target}.3"

                    if not os.path.exists(temp_file):
                        client.retrieve(dataset, request, temp_file)

                    cmd = [
                        "ncwa",
                        "-a",
                        "forecastMonth",
                        temp_file,
                        temp_file1,
                    ]
                    if not os.path.exists(temp_file1):
                        subprocess.run(cmd, check=True)

                    cmd = [
                        "ncpdq",
                        "-a",
                        f"{time_name},number,latitude,longitude",
                        temp_file1,
                        temp_file2,
                    ]
                    if not os.path.exists(temp_file2):
                        subprocess.run(cmd, check=True)

                    cdo.remapcon(grid_file, input=temp_file2, output=temp_file3)
                    shutil.move(temp_file3, target)
                    os.remove(temp_file)
                    os.remove(temp_file1)
                    os.remove(temp_file2)


@app.command()
def process_c3s_hist(start_year: int, end_year: int, data_root: str, grid_file: str):
    C3S().download_hist(start_year, end_year, data_root, grid_file)


if __name__ == "__main__":
    app()
