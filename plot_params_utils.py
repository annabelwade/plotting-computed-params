# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import json
import gcsfs
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %%
fp = r'C:\Users\annab\Desktop\ArgoEKE\github-argo-eke\data\pangeo-181919-e7bc5bdaf4d5.json'
with open(fp) as f:
    token = json.load(f)
gcs = gcsfs.GCSFileSystem(token=token) # file system instance

# %%
def load_polygons(fp):
	"""
	Input:
		fp: filepath to .shp file containing polygon geometries
			
	Returns:
		list of shapely.geometry.Polygon objects
	"""
	import pickle

	with open(fp, 'rb') as f:
		loaded_lst = pickle.load(f)
	return loaded_lst

def to_ignore(lat, lon, polygons):
	"""
	Input:
		lat: np array of latitudes ###xarray.DataArray of latitudes
		lon: np array of longitudes ### xarray.DataArray of longitudes
		polygons: list of shapely.geometry.Polygon objects
		
	Returns:
		numpy array of booleans, where True means the corresponding
		cycle's lat and lon fell into one of the polygons
	"""
	from shapely import prepare, contains_xy

	bools = np.zeros((len(polygons),lat.shape[0]))
	prepare(polygons) # is recommended to improve performance when using contains func
	for i in range(len(polygons)):
		bools[i,:] = contains_xy(polygons[i], lon, lat)
	return np.any(bools, axis=0)

# %%
def make_dir(dirname, clean=False):
    """
    Make a directory if it does not exist.

    Input:
		dirname: directory name string
		clean: boolean, whether to clobber/overwrite the directory if it already exists
    """
    import shutil
    import os
    
    if clean == True:
        shutil.rmtree(dirname, ignore_errors=True)
        os.mkdir(dirname)
    else:
        try:
            os.mkdir(dirname)
        except OSError:
            pass # assume OSError was raised because directory already exists

# %%
def map_fp(level_index, parameter_index, parameter_type, local=True):
    """
    Generates filepath for parameter map image according to pressure level and parameter type/index
    """
    import os

    if local:
        plot_dir = os.getcwd()
    else: # if not local, assume on GCS
        plot_dir = f'pangeo-argo-eke/global/plots'

    if parameter_type == 'beta':
        fp = plot_dir + r'/param_maps/betaparam_maps/betaparam{i}/beta_map_plevel{p}_param{i}.png'.format(p=level_index, i=parameter_index)
    elif parameter_type == 'covar':
        fp = plot_dir + r'/param_maps/covarparam_maps/covarparam{i}/covar_map_plevel{p}_param{i}.png'.format(p=level_index, i=parameter_index)
        # r'\small_scale_parameters\covarparam_maps\covar_param{i}\covar_map_plevel{p}_param{i}.png'.format(p=level_index, i=parameter_index)
    else:
        raise Exception('Invalid parameter type')
    return fp

# %%
def plot_gridded_param( parameter_type, level_index, parameter_index, ignore_regions=False ):
    """
    Takes in the type of parameter, 'beta' (large-scale) or 'covariance' (small-scale), 
    and the pressure index (int 0-28) and parameter index (int 0 - 9)
    and creates a global map

    ignore_regions: boolean, whether or not to ignore previously defined regions (seas, coastal, etc.)
    """
    import cartopy
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    # increase resolution of matplib plots
    plt.rcParams['figure.dpi'] = 450
    
    if parameter_type == 'beta':
        # Get beta parameter data at this pressure level
        GCSpath=f'pangeo-argo-eke/global/large_scale_parameters/with_pressure_coordinate/betaparameters_global_ws500_plevel{level_index}.zarr'
        GCSmapper = gcs.get_mapper(GCSpath)
        ds = xr.open_zarr(GCSmapper)
        gridded_data = ds.isel(parameter=parameter_index).betaparam
        # print(gridded_data)

    elif parameter_type == 'covariance':
        # Get covariance parameter data at this pressure level
        GCSpath=f'pangeo-argo-eke/global/small_scale_parameters/with_pressure_coordinate/covparameters_global_ws500_N24_plevel{level_index}.zarr'
        GCSmapper = gcs.get_mapper(GCSpath)
        ds = xr.open_zarr(GCSmapper)
        gridded_data = ds.isel(parameter=parameter_index).covparam

    else:
        raise Exception('Invalid parameter type')
    
    if ignore_regions:
        polygons = load_polygons(r'C:\Users\annab\Desktop\ArgoEKE\ignored_regions.shp')
        LON, LAT = np.meshgrid(gridded_data.longitude, gridded_data.latitude)
        bool_array = to_ignore(LAT.flatten(), LON.flatten(), polygons).reshape(LAT.shape)
        gridded_data = gridded_data.where(~bool_array)
    
    pres_str = 'pressure level: ' + r'$\bf{' + str(gridded_data.pressure.item()) +  '}$' + ' dbar'
    title = f'{parameter_type} parameter index: ' + r'$\bf{' +  f'{parameter_index}'  +  '}$' + f', {pres_str}'
    fig,ax = plt.subplots(figsize=(12,6),subplot_kw={'projection':ccrs.Robinson()})
    gridded_data.plot(ax=ax,transform=ccrs.PlateCarree(),cmap='viridis')
    ax.coastlines(); ax.set_title(title)
    plt.text(0.005, 1.1, 'window_size: 500 km', transform=ax.transAxes, fontsize=7, verticalalignment='top')
    plt.savefig(map_fp(level_index, parameter_index, parameter_type),facecolor='white',bbox_inches='tight'); plt.close()

# %%
def show_image(fp,local=True):
    """
    Displays image at filepath fp in interactive window
    """
    plt.rcParams["figure.figsize"] = (8,6)
    plt.rcParams['figure.dpi'] = 200
    plt.clf()
    if not local:
        fp = gcs.open(fp)
    img = mpimg.imread(fp)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# %%
def plot_params_tool_button():

    import ipywidgets as widgets
    from IPython.display import display, clear_output

    map_fp_lst = []

    for var in ['beta', 'covar', 'dh']:
        if var == 'beta':
            sublst =[]
            # Make list of pre-made images
            for param_i in range(0,10):
                subsublst = []
                for pres_i in range(0,29):
                    subsublst += [map_fp(pres_i, param_i, 'beta',local=False)]
                sublst += [subsublst]
            map_fp_lst += [sublst]
        elif var == 'covar':
            map_fp_lst += [] # TODO
        elif var == 'dh':
            map_fp_lst += [] # TODO

    button_values = ['Betaparameter', 'Covariance parameter', 'Dynamic Height']
    # Define custom style for the sliders
    slider_style = {'description_width': '50%', 'width': '600px', 'margin': '10px 0px', 'font_size': '40px'}

    slider1 = widgets.IntSlider(value=0, min=0, max=9, description='Betaparameter index')
    slider2 = widgets.IntSlider(value=0, min=0, max=28, description='Pressure level index')
    # Apply the custom style to the sliders
    slider1.layout = widgets.Layout(**slider_style); slider2.layout = widgets.Layout(**slider_style)
    slider1.style.description_width = '150px'; slider2.style.description_width = '150px'

    button = widgets.ToggleButtons(options=button_values, description='Field')
    button.style.button_width = '200px'

    image_out = widgets.Output()

    # Initialize the image with the first image from the list
    with image_out: 
        show_image(map_fp_lst[0][0][0],local=False) 

    def update_image(change):
        with image_out:
            # Clear the previous image
            clear_output(wait=True)

            # Retrieve the slider values
            index1 = slider1.value
            index2 = slider2.value
            button_value = button.value
            button_index = button_values.index(button_value)

            # Load the selected image
            image_path = map_fp_lst[button_index][index1][index2]
            show_image(image_path,local=False)

    # Register the update_image function as the callback for slider/button changes
    slider1.observe(update_image, names='value')
    slider2.observe(update_image, names='value')
    button.observe(update_image, names='value')

    # Create a VBox layout to display the widgets vertically
    widgets_box = widgets.VBox([slider1, slider2, button])
    # Display the widgets
    display(widgets_box, image_out)
