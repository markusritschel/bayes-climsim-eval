# %%
from src import *
import fire
from dask import delayed
from pathlib import Path
from my_code_base.core.xarray import compress_xarray
from my_code_base.esd.utils import compute_weighted_mean
import pandas as pd
import re
import pandas as pd
import xesmf as xe
import xarray as xr
from tqdm.auto import tqdm
import intake


from src import setup_logger
log = setup_logger()


#%% Define main functions
def process_observations():
    """Process the observations data by reading, cleaning, computing weighted mean,
    compressing, and saving it to a NetCDF file.
    """
    ds = read_hadcrut()
    ds_ = clean_coordinates(ds['tas_mean'].sel(time=slice("1880","2005")))
    ds_ = compute_weighted_mean(ds_)
    output_file = DATA_DIR/"processed/observations_1880-2005.nc"
    ds_ = compress_xarray(ds_, 8)
    ds_.to_netcdf(output_file)
    log.info(f"Saved observations to {output_file.relative_to(BASE_DIR)}")


def process_simulations(experiment_id):
    """Process simulations for a given experiment ID by reading, cleaning, computing weighted mean,
    compressing, and saving it to a NetCDF file.

    Parameters
    ----------
    experiment_id: str
        The ID of the experiment.
    """
    ds_obs = read_hadcrut()
    df = pd.read_csv(DATA_DIR/f"catalog-filter-{experiment_id}.csv")

    all_ds = {}

    for id, df_ in tqdm(df.groupby(['experiment', 'model', 'ensemble_member']), 
                        desc="Generate dataset dictionary"):
        ds_dict = generate_ds_dict(id, df_, ds_obs)
        all_ds.update(ds_dict.compute())


    all_ds = unify_datasets(all_ds)
    ds_comb = combine_datasets(all_ds)
    ds_comb = build_anomalies(ds_comb)
    ds_comb = mask_with_observations(ds_comb, ds_obs['tas_mean'])
    ds_comb = compute_weighted_mean(ds_comb)
    ds_comb = clean_coordinates(ds_comb)

    # xr.backends.file_manager.FILE_CACHE.clear()
    ds_comb = compress_xarray(ds_comb, 5)

    start, end = ds_comb.time.dt.year.values[0], ds_comb.time.dt.year.values[-1]
    output_file = DATA_DIR/f"processed/{experiment_id}_ensemble_{start}-{end}.nc"
    save_netcdf(ds_comb, output_file)
    log.info(f"Saved {experiment_id} simulations to {output_file.relative_to(BASE_DIR)}")
    return


#%% Define helper functions
def regrid(ds_in, regrid_to, method='bilinear'):
    """Regrids a given dataset to a target grid using the specified method.

    Parameters
    ----------
    ds_in: xarray.Dataset
        The input dataset to be regridded.
    regrid_to: xarray.Dataset
        The target grid to regrid the input dataset to.
    method: str (optional)
        The regridding method to use. Defaults to 'bilinear'.

    Returns
    -------
    xarray.Dataset
        The regridded dataset.
    """
    regridder = xe.Regridder(ds_in, regrid_to, method=method, periodic=True, ignore_degenerate=True)
    return regridder(ds_in)


    """Modify a dictionary holding a xarray.Dataset with piControl data by splitting it into sequences of length `periods` with an overlap of half the length."""
def split_picontrol(d: dict, periods=126, freq='1MS'):
    """Split the input dictionary of datasets into multiple ensemble members.

    Parameters
    ----------
    d: dict
        Input dictionary of datasets.
    periods: int
        Number of periods in each ensemble member.
    freq: str
        Frequency of the timestamps.

    Returns
    -------
    output_dict: dict
        Dictionary containing the split ensemble members.
    """
    timestamps = pd.date_range(start="1880-01-01", freq=freq, periods=periods)
    output_dict = {}

    i = 0
    (k, ds), = d.items()

    if ds.time.size < timestamps.size:
        log.warning(f' ├── Skip {k}. Time series too short!')
    while (da := ds.isel(time=slice((i*periods)//2, (i*periods//2)+periods))).time.size >= periods:
        da['time'] = timestamps
        k_ = re.sub(r"r(\d)(i\dp\d)", rf"r\g<1>{i+1:02d}\g<2>", k)
        output_dict[k_] = da
        i += 1
    log.debug(f" └── Created {i:2d} ensemble members from {k}")
    return output_dict


def mask_with_observations(ds, da_obs: xr.DataArray):
    log.info("Mask data with observations")
    return ds.where(~da_obs.isnull())


def build_anomalies(ds, ref_period=["1961","1990"]):
    """Build anomalies by subtracting the reference average from the actual values.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the actual values.
    ref_period: list (optional)
        The reference period to calculate the average over. Defaults to ["1961","1990"].

    Returns
    -------
    xarray.Dataset
        The dataset with anomalies calculated.
    """
    start, end = ref_period
    log.info("Build anomalies")
    log.info(f" ├── Build the monthly average for the reference period {start}–{end}")
    ref_monmean = ds.sel(time=slice(start,end)).groupby('time.month').mean()
    log.info(" └── Subtract the reference average from the actual values")
    return ds.groupby('time.month') - ref_monmean


def file_preproc(ds):
    ds = ds[['tas']]
    ds['time'] = xr.CFTimeIndex(ds.time.values, calendar="noleap")
    return ds


def unify_datasets(ds_dict):
    log.info("Unify datasets. This may take a while.")
    log.info("Rename and clean coordinates.")
    for k, ds in tqdm(ds_dict.items(), desc="Unify datasets"):
        # ds = ds.copy(deep=True).chunk(dict(time=-1, lat=ds.lat.size//8, lon=ds.lon.size//8))

        ds = unify_ds(ds)

        ds_dict[k] = ds

    return ds_dict


def unify_ds(ds):
    ds = unify_time(ds)
    try:
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    except:
        pass
    ds = clean_coordinates(ds)
    return ds


def unify_time(ds):
    """
    Preprocesses the time dimension of a given dataset.

    Parameters
    ----------
    ds: xarray.Dataset
        The input dataset.

    Returns
    -------
    xarray.Dataset
        The preprocessed dataset with a unified time dimension.
    """
    ds = ds.convert_calendar('noleap', align_on='date')

    ds = ds.drop_duplicates('time', keep='first')
    if ds.dropna(dim='time', how='all').time.size < 126*12:
        id_key = ds.attrs.get('member_id', 'data')
        log.warning(f"Omit {id_key}. Too short.")
    
    ds = ds.sel(time=slice("1880", "2005"))

    is_monotonic_increasing = xr.CFTimeIndex(ds.time.values).is_monotonic_increasing
    if not is_monotonic_increasing:
        log.error("NOT MONOTONIC INCREASING!")

    ds['time'] = pd.to_datetime(ds['time'].dt.strftime('%Y-%m-01'))
    ds = ds.sortby('time')

    return ds


def rename_coordinates(ds):
    try:
        ds = ds.rename({'longitude':'lon', 'latitude':'lat'})
    except:
        pass
    return ds


def combine_datasets(ds_dict):
    """Combine multiple datasets into a single dataset with a `member` coordinate.

    Parameters
    ----------
    ds_dict: dict
        A dictionary containing the datasets to be combined. The keys represent the member names, 
        and the values represent the datasets.

    Returns
    -------
    ds_comb: xarray.Dataset
        The combined dataset with a `member` coordinate.
    """
    log.info("Combine all datasets in a single dataset with `member` coordinate.")
    ds_comb = xr.concat(
        list(ds_dict.values()),
        pd.Index(list(ds_dict.keys()), name="member"),
        compat="override",
        coords="minimal",
        combine_attrs="override",
    )
    return ds_comb


def clean_coordinates(da):
    return da.drop_vars([c for c in da.coords if c not in da.dims])


def read_hadcrut():
    """Reads HadCRUT observations from an intake catalog and performs necessary preprocessing.

    Returns
    -------
    ds_obs: xarray.Dataset
        Preprocessed HadCRUT observations dataset.
    """
    cat_file = DATA_DIR/"obs-intake.yaml"
    log.info(f"Read HadCRUT observations from intake catalog {cat_file.relative_to(BASE_DIR)}")
    cat_obs = intake.open_catalog(cat_file)
    ds_obs = cat_obs.HadCRUT5.to_dask()
    ds_obs = ds_obs.rename({'longitude': 'lon', 'latitude':'lat'})
    ds_obs = unify_time(ds_obs)
    return ds_obs


def load_dataset(files):
    """Load and preprocess a dataset from multiple files.

    Args
    ----
    files: list
        A list of file paths to load the dataset from.

    Returns
    -------
    xr.Dataset
        The loaded and preprocessed dataset.
    """
    log.debug(f" ({len(files)} files)")
    ds = xr.open_mfdataset(files, 
                           use_cftime=True, chunks={},
                           compat='override', coords='minimal', combine='nested', concat_dim='time',
                           parallel=True, 
                           preprocess=file_preproc
                           )
    return ds


@delayed
def generate_ds_dict(id, df_, ds_obs):
    # log = logging.getLogger(__name__)
    # log.setLevel(logging.WARNING)
    experiment, model, ensemble_member = id
    id_key = '.'.join(id)

    # if experiment != 'piControl':
    #     return {}

    log.debug(id_key)

    df_ = df_.sort_values('version').drop_duplicates(['temporal_subset'], keep='last')

    files = df_.sort_values('temporal_subset')['uri'].values
    ds = load_dataset(files)
    ds.attrs['id'] = id_key

    ds = regrid(ds, ds_obs)
    ds_dict = {id_key: ds}

    if experiment == 'piControl':
        ds_dict = split_picontrol(ds_dict, periods=126*12)

    return ds_dict


def save_netcdf(ds, output_file):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_file)
    return


if __name__ == '__main__':
    fire.Fire({'sim': process_simulations, 'obs': process_observations})
