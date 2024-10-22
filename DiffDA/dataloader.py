import itertools
import os
from typing import Iterable, Iterator, Sequence, Any, Mapping, Union, Optional
import jax_dataloader
import xarray as xr
import numpy as np
from jax_dataloader import Dataset
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import cKDTree
import pyinterp
import pyinterp.backends.xarray

# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def _get_nearest_idx(array, values, periodic_lon=False):
    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    if array[-1] < array[0]:
        array = array[::-1]
        flip = True
    else:
        flip = False
    if periodic_lon:
        array = np.append(array, 360)

    assert (array.min() <= values.min()) and (array.max() >= values.max()), print(f"Grid min: {array.min()}, Grid max: {array.max()}, Value min: {values.min()}, Value max: {values.max()}")

    idxs = np.searchsorted(array, values, side="left")
    
    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1
    #nearest_values = array[idxs]
    if flip:
        idxs = len(array) - 1 - idxs
    if periodic_lon:
        idxs[idxs==len(array)-1] = 0
    return idxs


class ENS10ERA5Dataset(Dataset):
    def __init__(self, ens10_path, era5_path, length=None, shuffle=True):
        self.ens10_ds = xr.open_zarr(ens10_path)
        self.era5_ds = xr.open_zarr(era5_path)
        assert len(self.ens10_ds.time) == len(self.era5_ds.time)
        if shuffle:
            self.ts = np.random.permutation(len(self.era5_ds.time))
        else:
            self.ts = np.arange(len(self.era5_ds.time))
        slice_example = np.squeeze(self.era5_ds.z.isel(time=0).to_numpy())
        self.shape = tuple(slice_example.shape)
        self.length = min(length, len(self.ens10_ds.time)) if length is not None else len(self.ens10_ds.time)
        self.num_ensemble = self.ens10_ds.dims['number']

    def __len__(self):
        return self.length*self.num_ensemble
    
    def getresolution(self):
        return self.shape
    
    def __getitem__(self, idx):
        try:
            idx = int(idx)
        except:
            pass
        if isinstance(idx, int):
            idx = idx % self.length
            ie = idx // self.length
            ens10_slice = self.ens10_ds.z.isel(time=self.ts[idx], number=ie).to_numpy().squeeze()
            era5_slice = self.era5_ds.z.isel(time=self.ts[idx]).to_numpy().squeeze()
            time = self.era5_ds.time[self.ts[idx]]
            dayofyear = float(time.dt.dayofyear) / 365
            hourofday = float(time.dt.hour) / 24
            assert len(ens10_slice.shape) == 2
            assert len(era5_slice.shape) == 2
            batch_dict = {"ens10": ens10_slice[np.newaxis, np.newaxis, :, :].astype(np.float32), "era5": era5_slice[np.newaxis, np.newaxis, :, :].astype(np.float32), 
                            "dist_frac": np.array([1.0], dtype=np.float32),
                            "dayofyear": np.array([dayofyear], dtype=np.float32),
                            "hourofday": np.array([hourofday], dtype=np.float32)}
            return batch_dict
        else:
            batches = [self.__getitem__(i) for i in idx]
            return {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0].keys()}
        
class GraphCastDiffusionDataset(Dataset):
    def __init__(self,
                 graphcast_pred_path: str,
                 weatherbench2_path: str,
                 climatology_path: Optional[str] = None,
                 downsample: bool = False,
                 sample_slice: Union[slice, None] =None,
                 offset: int = 9, # (48/6 + 1), 48h is the lead time, +1 for 2 step prediction
                 num_sparse_samples: int = 0,
                 blur_kernel_size: int = 3,
                 fixed_measurements: bool = False,
                 disable_pred_offset_check: bool = False,
                 observation_path: Optional[str] = None) -> None:
        """
        @param graphcast_pred_path: Path to one year of predictions
        @param weatherbench2_path: Path to one year of groundtruth data
        @param downsample:
        @param sample_slice: If None, uses the whole year to train, else uses a slice into the predictions. Assumes step is 1 (or None).
        @param offset: Offset for the slice of the prediction w.r.t. to the ground truth.
        @param num_sparse_samples:
        @param blur_kernel_size:
        """
        self.ds_gc = xr.open_zarr(graphcast_pred_path)
        self.ds_wb = xr.open_zarr(weatherbench2_path)

        self.ds_gc = self.ds_gc.drop_vars('time')#.rename(datetime='time')
        self.ds_wb = self.ds_wb.rename(time='datetime')
        static_fields = ['land_sea_mask', 'geopotential_at_surface']
        self.ds_static = self.ds_wb[static_fields]
        self.ds_wb = self.ds_wb.drop_vars(static_fields)

        self.offset = offset
        if not disable_pred_offset_check:
            assert self.ds_wb.datetime[self.offset] == self.ds_gc.datetime[0], f"Offset {self.offset} is not compatible with the data: {self.ds_wb.datetime[self.offset]} != {self.ds_gc.datetime[0]}"
        if sample_slice is not None:
            assert sample_slice.step == 1 or sample_slice.step is None
            # We currently can't use the last offset days of the year
            assert offset + sample_slice.stop < len(self.ds_wb.datetime)

            self.ds_gc = self.ds_gc.isel(datetime = sample_slice)
            # Offset the GT slice
            self.ds_wb = self.ds_wb.isel(datetime = slice(offset + sample_slice.start, offset + sample_slice.stop))

            assert len(self.ds_wb.datetime) == len(self.ds_gc.datetime)

        else:
            # slice the wb GT by the offset from the start and the gc prediction from the end
            self.ds_gc = self.ds_gc.isel(datetime = slice(0, len(self.ds_wb.datetime) - self.offset))
            self.ds_wb = self.ds_wb.isel(datetime = slice(self.offset, None))

            assert len(self.ds_wb.datetime) == len(self.ds_gc.datetime)

        if not disable_pred_offset_check:
            assert self.ds_wb.datetime[0] == self.ds_gc.datetime[0]

        self.length = min(len(self.ds_wb.datetime), len(self.ds_gc.datetime))

        rename_dict = dict(latitude='lat', longitude='lon')
        self.ds_wb = self.ds_wb.rename(rename_dict)
        self.ds_static = self.ds_static.rename(rename_dict)

        if climatology_path is not None:
            self.ds_clim = xr.open_zarr(climatology_path).rename(rename_dict)
        else:
            self.ds_clim = None

        self.downsample = downsample
        self.observation_path = observation_path

        if observation_path is not None:
            self.num_sparse_samples = -1
            self.use_real_obs = True
        else:
            self.num_sparse_samples = num_sparse_samples
            self.use_real_obs = False
            self.coords_latlon = None
        self.fixed_measurements = fixed_measurements
        self.mask = None
        self.coords_xyz = None
        if self.num_sparse_samples > 0:
            if downsample:
                nlat = 181
                nlon = 360
            else:
                nlat = 721
                nlon = 1440
            lat = np.linspace(-90, 90, nlat)
            lon = np.linspace(0, 360, nlon)
            lat, lon = np.meshgrid(lat, lon, indexing='ij')
            self.grid_lonlat = np.stack([lon, lat], axis=-1)
            lat = np.deg2rad(lat)
            lon = np.deg2rad(lon)
            z = np.sin(lat)
            y = np.cos(lat) * np.sin(lon)
            x = np.cos(lat) * np.cos(lon)
            grid_xyz = np.stack([x, y, z], axis=-1)
            assert grid_xyz.shape == (nlat, nlon, 3)
            self.grid_xyz = grid_xyz
            self.blur_kernel_size = blur_kernel_size


    def __len__(self):
        return self.length
    
    def _generate_sparse_samples(self, num_sparse_samples, grid_lat, grid_lon):
        if self.fixed_measurements and self.mask is not None:
            return self.mask, self.coords_xyz, self.coords_latlon
        #import scipy.ndimage
        nlat = len(grid_lat)
        nlon = len(grid_lon)
        if self.use_real_obs:
            idx_lat = _get_nearest_idx(grid_lat, self.coords_latlon[:, 1])
            idx_lon = _get_nearest_idx(grid_lon, self.coords_latlon[:, 0], periodic_lon=True)
        else:
            idx_lat = np.random.randint(0, nlat, size=num_sparse_samples)
            idx_lon = np.random.randint(0, nlon, size=num_sparse_samples)
        coords_latlon = np.stack([grid_lat[idx_lat], grid_lon[idx_lon]], axis=-1)
        mask = np.zeros((nlat, nlon), dtype=np.float32)
        mask[idx_lat, idx_lon] = 1.0
        coords_latlon_rad = coords_latlon * np.pi / 180
        coords_xyz = np.stack([np.cos(coords_latlon_rad[:, 0]) * np.cos(coords_latlon_rad[:, 1]), 
                               np.cos(coords_latlon_rad[:, 0]) * np.sin(coords_latlon_rad[:, 1]), 
                               np.sin(coords_latlon_rad[:, 0]) ], axis=-1)
        if self.blur_kernel_size > 0:
            radius = int(4*self.blur_kernel_size)+1
            assert radius > 0
            kernel_size = 2*radius+1
            idx1, idx2 = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), indexing='ij')
            mask_smooth = np.zeros((nlat + 2*radius, nlon + 2*radius), dtype=np.float32)
            lat_pad = np.pad(grid_lat.data, (radius, radius), mode='edge')
            for i, j in zip(idx_lat, idx_lon):
                scale = np.cos(np.deg2rad(lat_pad[i:i+kernel_size])) + 0.05
                scale = scale.reshape((-1, 1))
                kernel = np.exp(-(idx1**2 + (idx2*scale)**2) / (2*self.blur_kernel_size**2))
                slice_ij = mask_smooth[i:i+kernel_size, j:j+kernel_size]
                mask_smooth[i:i+kernel_size, j:j+kernel_size] = np.maximum(slice_ij, kernel)
            mask = mask_smooth[radius:-radius, radius:-radius]
            #mask = scipy.ndimage.gaussian_filter(mask, sigma=self.blur_kernel_size, radius=radius)
            #mask = mask / np.max(mask)
            #mask[mask < 1e-2] = 0.0
        #coords_xyz = np.random.randn(num_sparse_samples, 3)
        #coords_xyz = coords_xyz / np.linalg.norm(coords_xyz, ord=2, axis=1, keepdims=True)
        #lat = np.arcsin(coords_xyz[:, 2]) * 180 / np.pi
        #lon = np.arctan2(coords_xyz[:, 1], coords_xyz[:, 0]) * 180 / np.pi
        #coords_latlon = np.stack([lat, lon], axis=1)
        if self.fixed_measurements:
            self.mask = mask
            self.coords_xyz = coords_xyz
            self.coords_latlon = coords_latlon
        return mask, coords_xyz, coords_latlon
    
    def _interpolate_slice(self, ds_slice, coords_xyz):
        eps1 = 0.1
        eps2 = 1e-4
        for attempt in range(10):
            try:
                # we failed all the attempts - deal with the consequences.
                coords_xyz2 = np.concatenate([coords_xyz * 2.0, coords_xyz*(1+eps1), coords_xyz*(1-eps2)], axis=0)
                ds_slice2 = np.concatenate([ds_slice, ds_slice, ds_slice], axis=0)
                ds_interp = griddata(coords_xyz2, ds_slice2,
                                    (self.grid_xyz[:,:,0], self.grid_xyz[:,:,1], self.grid_xyz[:,:,2]), 
                                    method='linear')
                # TODO: DIVAnd.jl for interp
                assert not np.any(np.isnan(ds_interp))
            except:
                # perhaps reconnect, etc.
                print(f"Interpolation attempt {attempt} failed. Trying again.")
            else:
                break
        else:
            raise RuntimeError("Interpolation failed.")
        return ds_interp
    
    def _interpolate_slice_pyinterp(self, ds_slice, coords_lonlat, covariance='matern_12', alpha=600_000, k=9):
        mesh = pyinterp.RTree()
        mesh.packing(coords_lonlat, ds_slice)
        field, neighbors = mesh.universal_kriging(
            self.grid_lonlat.reshape((-1, 2)),
            within=False,  # Extrapolation is forbidden
            radius=2_000_000, # in meters
            k=k,
            covariance=covariance,
            alpha=alpha,
            num_threads=8)
        field = field.reshape((self.grid_lonlat.shape[0], self.grid_lonlat.shape[1]))
        #if fill_nan:
        #    field = np.where(np.isnan(field), 0, field)
        assert not np.any(np.isnan(field))
        return field
    
    def _interpolate_2dgrid_to_sparse_pyinterp(self, ds, lon, lat):
        interpolator = pyinterp.backends.xarray.Grid2D(ds)
        ds_sparse = interpolator.bivariate(
            coords=dict(lon=lon, lat=lat)
            )
        return ds_sparse
    
    def _interpolate_grid_to_sparse_pyinterp(self, ds, lon, lat):
        da_sparse_list = []
        for var in ds.data_vars:
            da = ds[var]
            if 'level' in da.dims:
                da_sparse_leveli_list = []
                for i in range(len(da.level)):
                    da_level = da.isel(level=i)
                    da_sparse = self._interpolate_2dgrid_to_sparse_pyinterp(da_level, lon, lat)
                    da_sparse_leveli_list.append(da_sparse)
                da_sparse = np.stack(da_sparse_leveli_list, axis=0)
                da_sparse = xr.DataArray(da_sparse, dims=['level', 'sparse_sample'], name=var, coords={'level': da.level})
                da_sparse_list.append(da_sparse)
            else:
                da_sparse = self._interpolate_2dgrid_to_sparse_pyinterp(da, lon, lat)
                da_sparse = xr.DataArray(da_sparse, dims=['sparse_sample'], name=var)
                da_sparse_list.append(da_sparse)
        ds_sparse = xr.merge(da_sparse_list)
        return ds_sparse
    
    def _interpolate_sparse_samples(self, ds: xr.Dataset, coords_latlon, ds_clim=None):
        if self.blur_kernel_size == 0:
            ds_out = ds.copy()
            return ds_out
        # TODO: only interpolate values within the mask
        ds_out = xr.zeros_like(ds)
        lat = coords_latlon[:, 0]
        lon = coords_latlon[:, 1]
        coords_lonlat = coords_latlon[:, ::-1]

        if ds_clim is not None:
            ds_sparse = self._interpolate_grid_to_sparse_pyinterp(ds - ds_clim, lon, lat)
        else:
            ds_sparse = self._interpolate_grid_to_sparse_pyinterp(ds, lon, lat)

        for var in ds.data_vars:
            da = ds[var]
            da_sparse = ds_sparse[var]
            if 'level' in da.dims:
                dims = ['level', 'lat', 'lon']
                #grid_interp = np.stack([self._interpolate_slice(da_samples.isel(level=i).values, coords_xyz) for i in range(len(da.level))], axis=0)
                da_leveli_interp_list = []
                for i in range(len(da.level)):
                    da_sparse_leveli = da_sparse.isel(level=i)
                    da_leveli_interp = self._interpolate_slice_pyinterp(da_sparse_leveli.values, coords_lonlat)
                    da_leveli_interp_list.append(da_leveli_interp)
                da_interp = np.stack(da_leveli_interp_list, axis=0)
            else:
                dims = ['lat', 'lon']
                da_interp = self._interpolate_slice_pyinterp(da_sparse.values, coords_lonlat)
            ds_out[var] += xr.DataArray(da_interp, dims=dims)
        if ds_clim is not None:
            ds_out += ds_clim
            ds_out = ds_out.drop_vars(['dayofyear', 'hour'])
        return ds_out
    
    def _interpolate_sparse_samples_idw(self, ds: xr.Dataset, coords_xyz, coords_latlon, neighbors=5):
        # http://earthpy.org/interpolation_between_grids_with_ckdtree.html
        if self.blur_kernel_size == 0:
            ds_out = ds.copy()
            return ds_out
        # TODO: only interpolate values within the mask
        ds_out = xr.zeros_like(ds)
        lat = xr.DataArray(coords_latlon[:, 0], dims=['sparse_sample'])
        lon = xr.DataArray(coords_latlon[:, 1], dims=['sparse_sample'])

        kdtree = cKDTree(coords_xyz)
        eps = 1e-7
        d, inds = kdtree.query(self.grid_xyz.reshape((-1, 3)), k = neighbors)
        dg = 2*np.arcsin(d*0.5) # Convert string length to great circle length
        weights = np.expand_dims(1.0 / (dg**2 + eps), axis=0)
        inv_weights_sum = 1.0 / np.sum(weights, axis=-1)

        for var in ds.data_vars:
            da = ds[var]
            da_samples = da.interp(lat=lat, lon=lon, method='nearest')
            if 'level' in da.dims:
                da_samples = da_samples.transpose('sparse_sample', 'level')
                nlevel = len(da.level)
            else:
                nlevel = 1
            assert coords_xyz.shape[0] == da_samples.shape[0]
            grid_interp = (da_samples.values.reshape((nlevel, -1))[:, inds] * weights).sum(axis=-1) * inv_weights_sum
            grid_interp = grid_interp.reshape(da.values.shape)
            if 'level' in da.dims:
                dims = ['level', 'lat', 'lon']
            else:
                dims = ['lat', 'lon']
            ds_out[var] += xr.DataArray(grid_interp, dims=dims)
        return ds_out
    
    def _disable_interpolation(self):
        if self.blur_kernel_size != 0:
            self.blur_kernel_size_ = self.blur_kernel_size
            self.blur_kernel_size = 0

    def _enable_interpolation(self, blur_kernel_size = None):
        if blur_kernel_size is not None:
            self.blur_kernel_size = blur_kernel_size
        else:
            self.blur_kernel_size = self.blur_kernel_size_
    
    def __getitem__(self, idx):
        try:
            idx = int(idx)
        except:
            pass
        if isinstance(idx, int):
            gc_slice = self.ds_gc.isel(datetime=idx)
            if gc_slice.lat[0].item() > gc_slice.lat[-1].item():
                gc_slice = gc_slice.isel(lat=slice(None, None, -1)) # the latitude should be in ascending order
            wb_slice = self.ds_wb.isel(datetime=idx).isel(lat=slice(None, None, -1))
            gc_slice = gc_slice.assign_coords(datetime = wb_slice.datetime.values)
            ds_static = self.ds_static.isel(lat=slice(None, None, -1))
            if self.downsample:
                wb_slice = wb_slice.isel(lon=slice(None, None, 4)).interp(lat=np.linspace(-90, 90, 181))
                gc_slice = gc_slice.isel(lon=slice(None, None, 4)).interp(lat=np.linspace(-90, 90, 181))
                ds_static = ds_static.isel(lon=slice(None, None, 4)).interp(lat=np.linspace(-90, 90, 181)) #.expand_dims('batch')
            extra_dict = {}
            if self.num_sparse_samples > 0:
                if self.ds_clim is not None:
                    clim_slice = self.ds_clim.sel(
                        hour=wb_slice.datetime.dt.hour.item(), 
                        dayofyear=wb_slice.datetime.dt.dayofyear.item(),
                        lat=wb_slice.lat,
                        lon=wb_slice.lon,
                        level=wb_slice.level)
                else:
                    clim_slice = None

                mask, coords_xyz, coords_latlon = self._generate_sparse_samples(self.num_sparse_samples, wb_slice.lat, wb_slice.lon)
                mask = xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': wb_slice.lat, 'lon': wb_slice.lon})
                mask = mask.expand_dims(['batch'], axis=0).to_dataset(name='mask')
                if "toa_incident_solar_radiation" in wb_slice.data_vars:
                    wb_slice_no_tisr = wb_slice.drop_vars(['toa_incident_solar_radiation'])
                else:
                    wb_slice_no_tisr = wb_slice
                wb_interp_slice = self._interpolate_sparse_samples(wb_slice_no_tisr.compute(), coords_latlon, ds_clim=clim_slice)
                #wb_interp_slice = self._interpolate_sparse_samples_idw(wb_slice.compute(), coords_xyz, coords_latlon) # Not working now!

                wb_interp_slice = wb_interp_slice.expand_dims(['batch', 'time'], axis=[0, 1])
                assert "toa_incident_solar_radiation" not in wb_interp_slice.data_vars
                extra_dict = {"mask": mask, 'weatherbench_interp': wb_interp_slice, 'obs_coords': np.expand_dims(coords_latlon, axis=0)}
            wb_slice = wb_slice.expand_dims(['batch', 'time'], axis=[0, 1])
            gc_slice = gc_slice.expand_dims(['batch', 'time'], axis=[0, 1])
            wb_slice = wb_slice.assign_coords(datetime=('time', wb_slice.datetime.data.reshape(wb_slice.time.data.shape)))
            gc_slice = gc_slice.assign_coords(datetime=('time', gc_slice.datetime.data.reshape(gc_slice.time.data.shape)))
            if "toa_incident_solar_radiation" not in wb_slice.data_vars:
                from graphcast.data_utils import add_tisr_var
                add_tisr_var(wb_slice)
            batch_dict = {"graphcast": gc_slice.compute(),
                          "weatherbench": wb_slice.compute(),
                          "static": ds_static.compute()}
            batch_dict.update(extra_dict)
            return batch_dict
        elif len(idx) > 1:
            batches = [self.__getitem__(i) for i in idx]
            wb_slices = xr.concat([b['weatherbench'] for b in batches], dim='batch')
            gc_slices = xr.concat([b['graphcast'] for b in batches], dim='batch')
            ds_static = batches[0]['static']
            extra_dict = {}
            if self.num_sparse_samples > 0:
                masks = xr.concat([b['mask'] for b in batches], dim='batch')
                wb_interp_slices = xr.concat([b['weatherbench_interp'] for b in batches], dim='batch')
                obs_coords = np.concatenate([b['obs_coords'] for b in batches], axis=0)
                extra_dict = {"mask": masks, "weatherbench_interp": wb_interp_slices, 'obs_coords': obs_coords}
            batch_dict = {"graphcast": gc_slices,
                          "weatherbench": wb_slices,
                          "static": ds_static}
            batch_dict.update(extra_dict)
            return batch_dict
        else:
            raise NotImplementedError

class DiffusionDataLoader(Iterable):
    samples_for_year: Mapping[int, slice]
    datasets: Sequence[GraphCastDiffusionDataset]
    size: int
    def __init__(self,
                 base_prediction_template: str,
                 weather_bench_template: str,
                 samples_for_year: Mapping[int, slice],
                 climatology_path: Optional[str] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 downsample: bool = True,
                 offset: int = 9,
                 disable_pred_offset_check: bool = False):
        """
        @param base_prediction_template A template string with a placeholder for the year. Correspond to the graphcast
        inference results.
        @param weather_bench_template A template string with a placeholder for the year. Corresponds to the weatherbench
        ground truth data.
        @param samples_for_year: maps each year to a slice of samples (numbered from 0 consecutively)
        @param batch_size: Batch size
        @param shuffle: Shuffle each year's data loader
        @param drop_last: Drop the last batch in each year
        @param downsample: If true, downsample the resolution by 4x
        @param offset: Offset for the slice of the prediction w.r.t. to the ground truth.
        """
        self.samples_for_year = samples_for_year
        self.datasets = [
            GraphCastDiffusionDataset(base_prediction_template.format(y),
                                      weather_bench_template.format(y),
                                      climatology_path=climatology_path,
                                      downsample=downsample,
                                      sample_slice=s,
                                      offset=offset,
                                      disable_pred_offset_check=disable_pred_offset_check)
            for y, s in self.samples_for_year.items()
        ]
        self.data_loaders = [jax_dataloader.DataLoader(d, backend="jax", batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
                             for d in self.datasets]

        self.size = sum([len(d) for d in self.data_loaders])
        # Make sure all data loaders have the same size.
        for d in self.data_loaders:
            assert len(d) == len(self.data_loaders[0]), "All data loaders must have the same size."

    def __len__(self):
        return self.size

    def __iter__(self) -> Iterator[Any]:
        return itertools.chain(*[iter(d) for d in self.data_loaders])

def load_normalization(path: str, downsample: bool = False) -> (Dataset, Dataset, Dataset):
    diffs_stddev_by_level = xr.load_dataset(f"{path}/diffs_stddev_by_level.nc").compute()
    mean_by_level = xr.load_dataset(f"{path}/mean_by_level.nc").compute()
    stddev_by_level = xr.load_dataset(f"{path}/stddev_by_level.nc").compute()
    return diffs_stddev_by_level, mean_by_level, stddev_by_level

def load_model_checkpoint(path: str) -> "tuple[graphcast.ModelConfig, graphcast.TaskConfig, dict]":
    from graphcast import graphcast, checkpoint
    with open(path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    #print("Model description:\n", ckpt.description, "\n")
    #print("Model license:\n", ckpt.license, "\n")

    return model_config, task_config, ckpt.params