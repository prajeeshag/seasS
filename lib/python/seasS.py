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

cdo = Cdo()

cds_client = cdsapi.Client()

NMME_MODELS = [
    "CanSIPS-IC4",
    "COLA-RSMAS-CESM1",
    "NCEP-CFSv2",
    "NASA-GEOSS2S",
    # "GFDL-SPEAR",
]


PERCENTILES = [10, 33, 66, 90]


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
    freq = {}
    for p in PERCENTILES:
        pname = f"p{p}"
        fname = f"f{p}"
        pvar = hist_ds[pname]
        fvar = var.copy(deep=True)
        if p < 50:
            fvar.values = var.values < pvar.values
        else:
            fvar.values = var.values > pvar.values
        if "M" in fvar.dims:
            fvar = fvar.sum(dim="M")
        else:
            fvar = fvar.sum(dim="number")

        fvar.name = fname
        freq[fname] = fvar

    fvar1 = freq["f33"].copy(deep=True)
    fvar1.values = fix_vectors(freq["f33"].values, freq["f66"].values)
    fvar1.name = "tercile_probability"
    fvar2 = freq["f10"].copy(deep=True)
    fvar2.values = fix_vectors(freq["f10"].values, freq["f90"].values)
    fvar2.name = "decile_probability"

    return ne, [fvar1, fvar2]


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
        "dwd": {
            "system": "21",
            "time_name": "forecast_reference_time",
            "prec_name": "tprate",
            "tref_name": "t2m",
        },
        "meteo_france": {
            "system": "9",
            "time_name": "forecast_reference_time",
            "prec_name": "tprate",
            "tref_name": "t2m",
        },
        "cmcc": {
            "system": "35",
            "time_name": "forecast_reference_time",
            "prec_name": "tprate",
            "tref_name": "t2m",
        },
        "ukmo": {
            "system": "603",
            "time_name": "indexing_time",
            "prec_name": "tprate",
            "tref_name": "t2m",
        },
        "ecmwf": {
            "system": "51",
            "time_name": "forecast_reference_time",
            "prec_name": "tprate",
            "tref_name": "t2m",
        },
    }
    FIELD_NAMES = {
        "prec": "total_precipitation",
        "tref": "2m_temperature",
    }
    PREFIX = "KMME"
    AREA = [40, 10, 0, 70]
    RANGES = ((1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5))

    def get_hist_pctl_path(
        self,
        model: str,
        month: int,
        lstart: int,
        lend: int,
        data_root: str,
        fldname: str,
    ):
        return (
            f"{data_root}/{self.PREFIX}/{fldname}"
            + f"/hist/{model}/percentiles/{month:02d}/"
            + f"L-{lstart}.5-{lend}.5/{model}-{fldname}-{month:02d}-L-{lstart}.5-{lend}.5_percentiles.nc"
        )

    def get_fcst_freq_path(self, accum_mon: int, data_root: str, fldname: str):
        return f"{data_root}/{fldname}/live/{fldname}-{self.PREFIX}-{accum_mon}monthly-frequencies.nc"

    def _compute_percentiles(self, in_file, fldname):
        ds = xr.open_dataset(in_file)
        da = ds[fldname]

        if len(da.shape) != 4:
            raise ValueError("Data array should be 4 dimensional")
        nt, ne, ny, nx = da.shape
        array = da.values.reshape(nt * ne, ny, nx)
        da_p10 = da[0, 0, :, :]
        da_p33 = da[0, 0, :, :]
        da_p66 = da[0, 0, :, :]
        da_p90 = da[0, 0, :, :]
        da_p33.values = np.nanpercentile(array, 33.3, 0)
        da_p66.values = np.nanpercentile(array, 66.6, 0)
        da_p10.values = np.nanpercentile(array, 10.0, 0)
        da_p90.values = np.nanpercentile(array, 90.0, 0)
        da_p10.name = "p10"
        da_p33.name = "p33"
        da_p66.name = "p66"
        da_p90.name = "p90"
        ds = xr.Dataset(
            {
                da_p33.name: da_p33,
                da_p66.name: da_p66,
                da_p10.name: da_p10,
                da_p90.name: da_p90,
            }
        )
        return ds

    def compute_percentiles(self, field_name, data_root):
        for lstart, lend in self.RANGES:
            for model, info in self.MODELS.items():
                for month in range(1, 13):
                    out_path = self.get_hist_pctl_path(
                        model, month, lstart, lend, data_root, field_name
                    )
                    print(f"Creating... {out_path}")
                    file_list = ""
                    for ll in range(lstart, lend + 1):
                        file_list = (
                            file_list
                            + " "
                            + self._get_hist_file_path(
                                field_name, data_root, model, month, ll + 1
                            )
                        )
                    temp_file = cdo.ensmean(input=file_list)
                    pctl_ds = self._compute_percentiles(
                        temp_file, info[f"{field_name}_name"]
                    )
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    pctl_ds.to_netcdf(out_path)

    def process_fcst(
        self,
        field_name: str,
        year: int,
        month: int,
        data_root: str,
    ):

        datasets = {}

        for lstart, lend in self.RANGES:
            for n, model in enumerate(NMME_MODELS):
                data_file = get_fcst_file_path(
                    year, month, lstart, lend, model, data_root, field_name
                )
                hist_pctl_data = get_hist_pctl_path(
                    model, month, lstart, lend, data_root, field_name
                )
                if n == 0:
                    ne, fr = compute_frequencies(data_file, hist_pctl_data, field_name)
                else:
                    nne, ffr = compute_frequencies(
                        data_file, hist_pctl_data, field_name
                    )
                    ne += nne
                    for i in range(len(fr)):
                        fr[i].values += ffr[i].values

            for n, (model, info) in enumerate(self.MODELS.items()):
                file_list = ""
                for ll in range(lstart, lend + 1):
                    file_list = (
                        file_list
                        + " "
                        + self._get_fcst_file_path(
                            field_name, data_root, model, year, month, ll + 1
                        )
                    )
                temp_file = cdo.ensmean(input=file_list)
                hist_pctl_file = self.get_hist_pctl_path(
                    model, month, lstart, lend, data_root, field_name
                )
                fldname = info[f"{field_name}_name"]
                nne, ffr = compute_frequencies(temp_file, hist_pctl_file, fldname)
                ne += nne
                for i in range(len(fr)):
                    fr[i].values += ffr[i].values

            for i in range(len(fr)):
                fr[i].values = fr[i].values / float(ne)

            accum_mon = lend - lstart + 1
            tstart = add_months(year, month, lstart)
            tend = add_months(year, month, lend)
            dataset = xr.Dataset({da.name: da for da in fr})
            dataset = dataset.drop_vars(["S"])
            dataset["tstart"] = (("S"), [tstart])
            dataset["tend"] = (("S"), [tend])
            if accum_mon in datasets:
                datasets[accum_mon].append(dataset)
            else:
                datasets[accum_mon] = [dataset]

        for accum_mon, dsets in datasets.items():
            dataset = xr.concat(dsets, dim="S")
            dataset = dataset.rename({"S": "time", "X": "lon", "Y": "lat"})
            ntime = len(dsets)
            ym = str(dataset["tstart"][0].values)
            dates = pd.date_range(
                start=f"{ym[0:4]}-{ym[4:6]}-01", periods=ntime, freq="MS"
            )
            ref_time = pd.Timestamp("1960-01-01")
            time_units = (dates - ref_time).days
            dataset.coords["time"] = ("time", time_units)
            dataset.coords["time"].attrs["units"] = "days since 1960-01-01"
            dataset.coords["time"].attrs["calendar"] = "gregorian"
            data_root1 = f"{data_root}/mme/freq/{year}/{month:02d}"
            output = self.get_fcst_freq_path(accum_mon, data_root1, field_name)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            if "forecast_reference_time" in dataset:
                dataset = dataset.drop_vars("forecast_reference_time")
            dataset.to_netcdf(output, unlimited_dims=["time"])
            print(f"created... {output}")

    def download_c3s_fcst(self, field_name, year, month, data_root, grid_file):
        for leadtime_month in map(str, list(range(2, 7))):
            self._download_preproc(
                field_name,
                year,
                year,
                grid_file,
                month,
                leadtime_month,
                data_root,
                hist=False,
            )

    def download_hist(
        self,
        field_name: str,
        start_year: int,
        end_year: int,
        data_root: str,
        grid_file: str,
    ):
        for month in map(str, list(range(1, 13))):
            for leadtime_month in map(str, list(range(2, 7))):
                self._download_preproc(
                    field_name,
                    start_year,
                    end_year,
                    grid_file,
                    month,
                    leadtime_month,
                    data_root,
                    hist=True,
                )

    def _download_preproc(
        self,
        field_name,
        start_year,
        end_year,
        grid_file,
        month,
        leadtime_month,
        data_root,
        hist,
    ):
        dataset = "seasonal-monthly-single-levels"
        for model, info in self.MODELS.items():
            if hist:
                target = self._get_hist_file_path(
                    field_name, data_root, model, month, leadtime_month
                )
            else:
                target = self._get_fcst_file_path(
                    field_name, data_root, model, start_year, month, leadtime_month
                )
            print(f"downloading and processing.... {target}")
            if os.path.exists(target):
                continue
            system = info["system"]
            if model == "ukmo" and start_year >= 2025 and month >= 3:
                system = "604"
            elif model == "dwd" and start_year >= 2025 and month >= 4:
                system = "22"
            time_name = info["time_name"]
            request = {
                "originating_centre": model,
                "system": system,
                "variable": [self.FIELD_NAMES[field_name]],
                "product_type": ["monthly_mean"],
                "year": list(map(str, list(range(start_year, end_year + 1)))),
                "month": [month],
                "leadtime_month": [leadtime_month],
                "data_format": "netcdf",
                "area": self.AREA,
            }
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            temp_file = f"{target}.download"
            temp_file1 = f"{target}.1"
            temp_file2 = f"{target}.2"
            temp_file3 = f"{target}.3"

            if not os.path.exists(temp_file):
                cds_client.retrieve(dataset, request, temp_file)

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

    def _get_hist_file_path(self, field_name, data_root, model, month, leadtime_month):
        return f"{data_root}/{self.PREFIX}/hist/{field_name}/{model}_{month}_{leadtime_month}.nc"

    def _get_fcst_file_path(
        self, field_name, data_root, model, year, month, leadtime_month
    ):
        return f"{data_root}/{self.PREFIX}/{field_name}/fcst/{year}/{model}_{field_name}_{year}_{month}_{leadtime_month}.nc"


@app.command()
def process_c3s_hist(
    field_name: str,
    start_year: int,
    end_year: int,
    data_root: str,
    grid_file: str,
):
    C3S().download_hist(field_name, start_year, end_year, data_root, grid_file)
    C3S().compute_percentiles(field_name, data_root)


@app.command()
def process_fcst(
    field_name: str,
    year: int,
    month: int,
    data_root: str,
):
    C3S().process_fcst(field_name, year, month, data_root)


@app.command()
def download_c3s_fcst(
    field_name: str,
    year: int,
    month: int,
    data_root: str,
    grid_file: str,
):
    C3S().download_c3s_fcst(field_name, year, month, data_root, grid_file)


if __name__ == "__main__":
    app()
