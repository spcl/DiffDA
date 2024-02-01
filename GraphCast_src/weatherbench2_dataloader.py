import xarray as xr
from copy import deepcopy

class WeatherBench2Dataset():
    def __init__(self, year: int, steps: int, steps_per_input:int = 3):
        self.vars = ["geopotential", "specific_humidity", "temperature",
                     "u_component_of_wind", "v_component_of_wind",
                     "vertical_velocity", "toa_incident_solar_radiation",
                     "10m_u_component_of_wind", "10m_v_component_of_wind",
                      "2m_temperature", "mean_sea_level_pressure",
                      "total_precipitation_6hr", "geopotential_at_surface", "land_sea_mask"]
        self.static_vars = ["geopotential_at_surface", "land_sea_mask"]
        self.ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr')
        self.ds = self.ds[self.vars]
        self.ds = self.ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        self.length = len(self.ds.time)
        self.steps = steps
        self.steps_per_input = steps_per_input
        self.coord_name_dict = dict(latitude="lat", longitude="lon")

    def __len__(self):
        num_batches = self.length // self.steps
        if self.length % self.steps < self.steps_per_input - 1:
            num_batches -= 1
        return num_batches

    def __getitem__(self, item):
        return self.get_data(item)

    def get_data(self, batch_idx):
        batch_idx = batch_idx % len(self)
        it_range = slice(batch_idx*self.steps, (batch_idx+1)*self.steps + self.steps_per_input - 1)
        static_data = self.ds[self.static_vars].rename(**self.coord_name_dict)
        data = self.ds.drop_vars(self.static_vars).isel(time=it_range)
        data = data.rename(**self.coord_name_dict)
        data = data.isel(lat =slice(None, None, -1))
        data = xr.merge([static_data, data.expand_dims({'batch':1})])
        data = data.assign_coords(datetime=(["batch", "time"], data.time.data.reshape(1, -1)))
        data = data.assign_coords(time=("time", data.time.data - data.time.data[0]))
        return data.compute()

if __name__ == "__main__":
    dataset = WeatherBench2Dataset(2016, steps=4, steps_per_input=3)
    print(len(dataset))
    data = dataset.get_data(-1)
    print(data)