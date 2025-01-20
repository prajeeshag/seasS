import calendar
import os
import shutil
from pathlib import Path

import requests
import typer
import xarray as xr

app = typer.Typer()

MODELS = [
    "CanSIPS-IC4",
    "COLA-RSMAS-CESM1",
    "GFDL-SPEAR",
    "NCEP-CFSv2",
    "NASA-GEOSS2S",
]

PERCENTILES = [10, 33, 66, 90]


def get_url(model: str, year: int, mon_abbr: str, lstart: int, lend: int):
    lead_range = f"L/{lstart}.5/{lend}.5/RANGEEDGES/"
    url1 = f"http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.{model}/"
    area = "Y/5/40/RANGE/X/20/60/RANGEEDGES"
    url2 = f".MONTHLY/.tref/{area}/S/%28%201%20{mon_abbr}%20{year}%29/VALUES/{lead_range}%5BL%5D/average/data.nc"
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


def get_hist_pctl_path(model: str, month: int, lstart: int, lend: int, root_path: str):
    return f"{root_path}/hist/{model}/percentiles/{month:02d}/L-{lstart}.5-{lend}.5/{model}-{month:02d}-L-{lstart}.5-{lend}.5_percentiles.nc"


def compute_frequencies(data_file, hist_pctl_data):
    ds = xr.open_dataset(data_file, decode_times=False)
    hist_ds = xr.open_dataset(hist_pctl_data)
    var = ds["tref"].squeeze()
    if len(var.shape) > 3 or len(var.shape) < 2:
        raise SystemError(f"var.shapre {var.shape}")
    ne = 1
    if len(var.shape) == 3:
        ne = var.shape[0]
    res = []
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
        res.append(fvar)
    return ne, res


def get_fcst_file_path(year, month, lstart, lend, model, root_path):
    data_file = f"{root_path}/fcst/{model}/averages/{year}/{month:02d}/L-{lstart}.5-{lend}.5/data.nc"
    Path(data_file).parent.mkdir(
        parents=True, exist_ok=True
    )  # Create directories if they don't exist
    return data_file


def get_mme_freq_path(year: int, month: int, lstart: int, lend: int, data_root: str):
    return f"{data_root}/MME/frequencies/{year}/{month:02d}/L-{lstart}.5-{lend}.5/MME-{month:02d}-L-{lstart}.5-{lend}.5_frequencies.nc"


@app.command()
def get_fcst_data(year: int, month: int, lstart: int, lend: int, data_root: str):
    mon_abbr = calendar.month_abbr[month]
    for model in MODELS:
        url = get_url(model, year, mon_abbr, lstart, lend)
        data_file = get_fcst_file_path(year, month, lstart, lend, model, data_root)
        print(f"downloading {data_file}")
        download_data(url, data_file)


@app.command()
def process_mme_freq(year: int, month: int, lstart: int, lend: int, data_root: str):
    for n, model in enumerate(MODELS):
        data_file = get_fcst_file_path(year, month, lstart, lend, model, data_root)
        hist_pctl_data = get_hist_pctl_path(model, month, lstart, lend, data_root)
        if n == 0:
            ne, fr = compute_frequencies(data_file, hist_pctl_data)
        else:
            nne, ffr = compute_frequencies(data_file, hist_pctl_data)
            ne += nne
            for i in range(len(fr)):
                fr[i].values += ffr[i].values
    for i in range(len(fr)):
        fr[i].values = fr[i].values / float(ne)
    dataset = xr.Dataset({da.name: da for da in fr})
    output = get_mme_freq_path(year, month, lstart, lend, data_root)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)


@app.command()
def mme_freq_path(year: int, month: int, lstart: int, lend: int, data_root: str):
    print(get_mme_freq_path(year, month, lstart, lend, data_root))


if __name__ == "__main__":
    app()
