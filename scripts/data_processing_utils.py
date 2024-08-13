# Import the necessary dependencies and initialize geemap before
# starting to use it. Also set file path

import os
import ee
import geemap
import geopandas as gpd
import math
import shapely
from shapely import affinity, LineString
from shapely.geometry import Point, Polygon, MultiPolygon, box, mapping
import pyproj
from shapely.ops import transform
import rasterio
from rasterio.mask import mask
import random
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
import dask_geopandas as dask_gpd
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

import warnings
from rasterio.features import rasterize
from shapely.geometry import shape, mapping


dir_path = os.path.abspath(os.path.dirname('__file__'))
tiff_path_rel = os.path.join(dir_path, 'GIS', 'dem')
tiff_path_abs = os.path.abspath(os.path.join(dir_path, tiff_path_rel))
shape_path_rel = os.path.join(dir_path, 'GIS', 'shapefiles')
shape_path_abs = os.path.abspath(os.path.join(dir_path, shape_path_rel))


tiff_path_cropped = os.path.join(dir_path, 'GIS', 'dem','cropped')
dir_path_local = os.path.abspath(os.path.join(dir_path, tiff_path_cropped))

save_img_epoch = os.path.join(dir_path, 'save_img_epoch')

class GISProcessor:
    def __init__(self):
        self.dir_path = os.path.abspath(os.path.dirname('__file__'))
        self.tiff_path_rel = os.path.join(self.dir_path, 'GIS', 'dem')
        self.tiff_path_abs = os.path.abspath(os.path.join(self.dir_path, self.tiff_path_rel))
        self.shape_path_rel = os.path.join(self.dir_path, 'GIS', 'shapefiles')
        self.shape_path_abs = os.path.abspath(os.path.join(self.dir_path, self.shape_path_rel))
        

    def get_valid_geometries(self,df): #USED IN SCRIPT

        # TODO: Check for methods to fix the invalid geometries instead of just removing them.

        df['is_valid'] = df.geometry.is_valid
        df = df[df['is_valid'] == True]
        df = df.drop(columns=['is_valid'])
        return df
    
    
    def dissolve_glaciersLEGACY(self,df, filter_for = None): 

        """
        Because the data is stored in a way that each glacier has multiple entries for each year, this function dissolves the data
        to have only one entry per glacier per year. The function can be used to filter for glaciers that still exist or have disappeared.

        # Deprecated. Updated version down below!
        """

         # check for valid geometries:

        invalid_geometries = df[~df.geometry.is_valid]
        if len(invalid_geometries) > 0:
            print(f"Warning: {len(invalid_geometries)} invalid geometries found in the input data. These will be removed.")
            df = self.get_valid_geometries(df)
        else:
            print("All geometries are valid.")   

        
        #get the year of the data:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df['year'] = df[col].dt.year
                break    

        # keep crs of dataframe:
        crs = df.crs

        if filter_for is not None:
            if filter_for == 'exists':
                df = df[df['glac_stat'] == 'exists']

                #dissolve the data:

                result_df = df.dissolve(by= ['glac_id', 'year'], aggfunc='first', as_index=False)
            
                """
                Alternative approach to dissolve the data:

                #dissolve the data:
                dissolved_df = df[['glac_id','year','geometry']].dissolve(by= ['glac_id', 'year'], 
                                                                    aggfunc='sum', as_index=False)
                
                    # Merge dissolved data back into the original DataFrame
                result_df = df[['line_type', 'glac_id', 'area', 'glac_name', 'glac_stat','year','src_date']].merge(
                    dissolved_df, on=['glac_id', 'year'], how='left') # Leads to geodataframe just beeing
                # a normal dataframe with geometry column. See:
                # https://geopandas.org/en/stable/docs/user_guide/mergingdata.html
                
                
                # turn df into gdf:
                result_df = gpd.GeoDataFrame(result_df, crs=crs, geometry='geometry')
                """

                # create a temporary helper column:
                result_df['glac_id_year_temp'] = result_df['glac_id'].astype(str) + '_' + result_df['year'].astype(str)

                # Delete duplicate entries:
                result_df = result_df.drop_duplicates(subset='glac_id_year_temp', keep = 'first')

                # Drop the helper column:
                result_df = result_df.drop(columns=['glac_id_year_temp'])
            
            elif filter_for == 'gone':
                df = df[df['glac_stat'] == 'gone']

                #dissolve the data:
                result_df = df.dissolve(by=['glac_id'], aggfunc='first', as_index=False)

                """
                Same alternative approach as above:
                dissolved_df = df[['glac_id','geometry']].dissolve(by= ['glac_id'], 
                                                                    aggfunc='sum', as_index=False)
                
                result_df = df[['line_type', 'glac_id', 'area', 'glac_name', 'glac_stat','year','src_date']].merge(
                    dissolved_df, on=['glac_id'], how='left') # Leads to geodataframe just beeing
                # a normal dataframe with geometry column. See:
                # https://geopandas.org/en/stable/docs/user_guide/mergingdata.html
                
                # turn df into gdf:
                result_df = gpd.GeoDataFrame(result_df, crs=crs, geometry='geometry')
                """

            else:
                warnings.warn("No known filter for glacier status provided. Function will stop")

                return

        else:
            warnings.warn("No filter for glacier status provided. Function will stop")

            return  
        
        result_df['year'] = result_df['year'].astype(int)
        result_df = result_df.drop(columns=['src_date'])

        return result_df   
    
    def dissolve_glaciers(self, df):

        """
        Function to dissolve glacier geometries in a GeoDataFrame. The function dissolves the geometries based on the glacier status.
        """

        # Fix the invalid geometries with a tolerance of 0.001
        df['is_valid'] = df.geometry.is_valid
        df.loc[~df['is_valid'], 'geometry'] = df.loc[~df['is_valid'], 'geometry'].apply(lambda geom: geom.simplify(tolerance=0.001))

        # Check if there are still invalid geometries
        df['is_valid'] = df.geometry.is_valid
        # If there are still invalid geometries, buffer the invalid geometries with a buffer of 0
        df.loc[~df['is_valid'], 'geometry'] = df.loc[~df['is_valid'], 'geometry'].apply(lambda geom: geom.buffer(0))

        df.drop(columns=['is_valid'], inplace=True) 

        invalid_geometries = df[~df.geometry.is_valid]
        if len(invalid_geometries) > 0:
            print(f'There are {len(invalid_geometries)} invalid geometries in the dataframe')
            df = self.get_valid_geometries(df)   

            print('Fixed the invalid geometries')
        else:
            print('No invalid geometries found')

        # Check for glacier status:
        all_exist = df['glac_stat'].isin(['exists']).all()
        all_gone = df['glac_stat'].isin(['gone']).all()

        if all_exist:
            # Make new helper column to group by for existing glaciers
            df['glac_id_year'] = df['glac_id'] + '_' + df['src_date'].dt.year.astype(str)    
            
            # Dissolve the dataframe
            df_new = df.dissolve(by='glac_id_year', as_index=False) 
            df_new.drop(columns=['glac_id_year'], inplace=True)
        elif all_gone:
            df_new = df.dissolve(by='glac_id', as_index=False)   
        else:
            raise ValueError('Gone and existing glaciers are mixed')

        # Check for empty geometries
        empty_geometries = df_new['geometry'].is_empty

        if empty_geometries.any():
            print(f'There are {empty_geometries.sum()} empty geometries in the dataframe')
            df_new = df_new[~empty_geometries]
            print('Removed the empty geometries')

        # Drop the helper as well as src_date coluns and make a new year column:
        df_new['year'] = df_new['src_date'].dt.year
        df_new.drop(columns=['src_date'], inplace=True)

        # Reset the index
        df_new.reset_index(drop=True, inplace=True)

        return df_new
    
    def extract_glaciers(self, df, glac_id = None, output_dir = None, to_file = None, iterate = None):

        """
        Take a single glacier or a whole dataframe of glaciers and extract the outlines into a single file.

        # Probably deprecated. Function has not been utilized yet
        """


        if to_file is None:
            print('Specify if you want to save to file (yes or no)')
            return

        if to_file == 'yes':

            if output_dir is None:
                print('Output directory not specified')
                return
            
            if glac_id is not None:
                df = df[df['glac_id'] == glac_id]
                grouped = df.groupby('year')
                for year, group_df in grouped:
                    group = grouped.get_group(year)
                    group.to_file(os.path.join(output_dir, f'{glac_id}_{year}.shp'))

            else:
                grouped = df.groupby(['glac_id', 'year'])
                for (glac_id, year), group_df in grouped:
                    group_df.to_file(os.path.join(output_dir, f'{glac_id}_{year}.shp'))
        elif to_file == 'no':
            if glac_id is not None:
                glac_dict = {}
                df = df[df['glac_id'] == glac_id]
                grouped = df.groupby('year')
                for year, group_df in grouped:
                    #group = grouped.get_group(year)
                    glac_dict[f'{glac_id}_{year}'] = grouped.get_group(year)

                return glac_dict
            else:
                """
                glac_dict = {}
                grouped = df.groupby(['glac_id', 'year'])
                for (glac_id, year), group_df in grouped:                
                    glac_dict[f'{glac_id}_{year}'] = group_df

                return glac_dict"""
                glac_dict = {}
                grouped = df.groupby('glac_id')
                for glac_id, group_df in grouped:
                    glac_dict[glac_id] = group_df
                return glac_dict
    
    def make_residuals(self, df, max_year = 2000, min_change = 10, max_change = 50): # USED IN SCRIPT
        """
        Function to calculate glacier residuals from a geodataframe containing glacier outlines.
        The function calculates the percentage change in area between the earliest and latest glacier record.
        If the percentage change is within the specified bounds, the residual is calculated and added to a new geodataframe.

        Due to the nature of the function, the new dataframe will contain polygons and multipolgons. Use the function
        keep_larges_multipolygon to convert the multipolygons to polygons.
        """
        # Filter out glaciers that are older than max_year
        df = df[df['year'] <= max_year]
        
        # Group by glacier id
        df_grouped = df.groupby('glac_id')

        # initialize new geodataframe
        new_gdf = gpd.GeoDataFrame(columns = ['glac_id',  'min_year', 'max_year', 'change in area','geometry'], crs = df.crs)
        
        # Iterate over DataFrame groups
        for name, _ in df_grouped:
            temp_df = df_grouped.get_group(name)
            min_year = temp_df.year.min()
            max_year = temp_df.year.max()

            # Get outline of earliest and latest glacier record
            earliest_outline = temp_df[temp_df.year == min_year].geometry
            latest_outline = temp_df[temp_df.year == max_year].geometry

            # Calculate area of earliest and latest glacier record
            area_earliest = earliest_outline.to_crs({'proj':'cea'}).area
            area_latest = latest_outline.to_crs({'proj':'cea'}).area

            # Calculate percentage change in area
            perc_change = ((area_earliest.values[0] - area_latest.values[0]) / area_earliest.values[0]) * 100
            
            # Check if percentage change is within specified bounds. If so, calculate residual and add to new geodataframe
            if perc_change > min_change and perc_change < max_change: 
                residual = earliest_outline.difference(latest_outline, align=False)
                residual_info = {
                    'glac_id': name,
                    'min_year': min_year,
                    'max_year': max_year,
                    'change in area': perc_change,
                    'geometry': residual
                }

                new_gdf = pd.concat([new_gdf, gpd.GeoDataFrame(residual_info)], ignore_index=True)

        return new_gdf 

    def keep_largest_multipolygon(self, multipolygon): #USED IN SCRIPT
        
        """
        Function to keep the largest polygon in a multipolygon geometry
        Use with apply() method on a GeoDataFrame

        df['geometry'] = df['geometry'].apply(keep_largest_multipolygon)
        
        Polygon and MultiPolygon needs to be imported from shapely.geometry module
        """

        if isinstance(multipolygon, MultiPolygon):
            areas = [poly.area for poly in multipolygon.geoms]
            return multipolygon.geoms[areas.index(max(areas))]
        elif isinstance(multipolygon, Polygon):
            return multipolygon

    def convert_tiff_dtype(self, input_folder, output_folder, new_dtype = 'unint16'):
        """
        Converts TIFF file data types in a folder, saving copies in a new folder.

        Args:
            input_folder: Path to the folder containing original TIFF files.
            output_folder: Path to the folder where converted copies will be saved.
            new_dtype: Desired data type for the converted files (e.g., 'uint8', 'uint16', 'float32').
        
        Probably deprecated. Corrected error in other function
            
        """

        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith('.tif') or filename.endswith('.tiff')  :
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                with rasterio.open(input_path) as src:
                    profile = src.profile.copy()
                    profile.update(dtype=new_dtype)

                    with rasterio.open(output_path, 'w', **profile) as dst:
                        for i in range(1, src.count + 1):
                            dst.write(src.read(i).astype(new_dtype), i)



class DatasetMaker:
    def __init__(self):
        self.dir_path = os.path.abspath(os.path.dirname('__file__'))
        self.tiff_path_rel = os.path.join(self.dir_path, 'GIS', 'dem')
        self.tiff_path_cropped = os.path.join(self.dir_path, 'GIS', 'dem', 'cropped')
        self.tiff_path_abs = os.path.abspath(os.path.join(self.dir_path, self.tiff_path_rel))

    def get_glac_area(self, df, glac_id):
        """
        Function to get the area of a glacier from a geodataframe containing glacier outlines.
        The function takes a glacier id and returns the area of the glacier.
        """            
        glac = df[df['glac_id'] == glac_id]
        area = glac['geometry'].to_crs({'proj':'cea'}).area.values[0] / 10**6
        return area
    
    def get_rectangle_area(self, gdf_bbox):
        """
        Function to get the area of a rectangle from a geodataframe containing a bounding box.
        """

        area = gdf_bbox['geometry'].to_crs({'proj':'cea'}).area.values[0] / 10**6
        return area
    
    def get_random_bbox_intersect(self, df_glacid, spatial_resolution = 30, img_res = 256, buffer = 5000, min_coverage = 0.1, max_coverage = None, max_tries = 100):

        """
        Applies, when area of glacier is larger than area of bbox

        Function which takes a single glac_id dataframe and returns a random bbox intersecting the polygon inside of the dataframe
        inside of a specific coverage. The function will keep generating random bboxes until the coverage is within the specified range.

        Also returns glac_id and intersection
        parameters:
        df_glacid : GeoDataFrame
            GeoDataFrame containing the glacier polygon
        buffer: int
            Buffer around the glacier polygon. Needed, so the bbox can move freely around the glacier polygon
        min_coverage: float
            Minimum coverage of the glacier polygon inside of the bbox
        max_coverage: float
            Maximum coverage of the glacier polygon inside of the bbox
        max_tries: int
            Maximum number of tries to generate a bbox with the specified coverage

        TODO: Maybe add iterations to generate multiple random bboxes (should not intersect), so I can enlarge dataset. 
        """
        
        height = img_res * spatial_resolution
        width = img_res * spatial_resolution
        
        # Ensure the glacier polygon is in EPSG:3857, so meter base coordinates
        df_glacid = df_glacid.to_crs('EPSG:3857')

        glac_id = df_glacid['glac_id'].iloc[0]

        # get the glacier polygon
        glacier_polygon = df_glacid['geometry'].iloc[0]

        minx, miny, maxx, maxy = glacier_polygon.buffer(buffer).bounds

        tries = 0

        while tries < max_tries:
            tries += 1
            
            # Generate random min coordinates within the bounds
            rand_minx = random.uniform(minx, maxx - width)
            rand_miny = random.uniform(miny, maxy - height)

            # Calculate max coordinates based on the width and height
            rand_maxx = rand_minx + width
            rand_maxy = rand_miny + height
            # Create a bounding box from the random coordinates
            rand_bbox = box(rand_minx, rand_miny, rand_maxx, rand_maxy)

            # Get the intersection of the glacier polygon and the random bbox
            intersection = rand_bbox.intersection(glacier_polygon) # grid_size=0

            # Calculate the percentage of the intersection
            coverage = intersection.area / rand_bbox.area

            # If the coverage is within the specified range, return the bbox
            if max_coverage is not None:
                if min_coverage <= coverage <= max_coverage:
                    break
            else:
                if min_coverage <= coverage:
                    break
                    
        if tries == max_tries:
            print('Failed to find a suitable bbox for glac_id: ', glac_id)
            return None, None

        else:    
            data_bbox = {'geometry': [rand_bbox], 'glac_id': [glac_id]}
            data_intersection = {'geometry': [intersection], 'glac_id': [glac_id]}
            gdf_bbox = gpd.GeoDataFrame(data_bbox, crs='EPSG:3857')
            gdf_intersection = gpd.GeoDataFrame(data_intersection, crs='EPSG:3857')
            return gdf_bbox, gdf_intersection
        
    def get_bbox_center(self, df_glacid, spatial_resolution = 30, img_res = 256):

        """
        Applies, when area of bbox is larger than area of glacier

        Function which takes a single glac_id dataframe and returns a random bbox with the center of the bbox
        beeing the center of the polygon
        """

        height = img_res * spatial_resolution
        width = img_res * spatial_resolution

        # Ensure, that dataframe is in EPSG:3857
        df_glacid = df_glacid.to_crs('EPSG:3857')

        # get glac_id of glacier
        glac_id = df_glacid['glac_id'].iloc[0]

        # get the glacier polygon
        glacier_polygon = df_glacid['geometry'].iloc[0]

        # get the center of the glacier polygon
        center = df_glacid['geometry'].centroid.iloc[0]

        # Get x and y of center
        x, y = center.x, center.y

        # Define coordinates for Rectangle
        minx = x - width / 2
        miny = y - height / 2
        
        maxx = x + width / 2
        maxy = y + height / 2

        # Make bbox
        bbox = box(minx, miny, maxx, maxy)

        # Get intersection of glacier polygon and bbox
        intersection = bbox.intersection(glacier_polygon) # grid_size=0

        # make intersection mask
        #intersection_mask = np.zeros((img_res, img_res), dtype=np.float32)
        #intersection_mask = rasterize([(glacier_polygon, 1)], out_shape=(img_res, img_res), transform=rasterio.transform.from_bounds(minx, miny, maxx, maxy, img_res, img_res), dtype=np.float32)
        
        # Create properties for GeoDataFrame
        data_bbox = {'geometry': [bbox], 'glac_id': [glac_id]}
        data_intersection = {'geometry': [intersection], 'glac_id': [glac_id]}
        

        # Transform into gdf
        gdf_bbox = gpd.GeoDataFrame(data_bbox, crs='EPSG:3857')
        gdf_intersection = gpd.GeoDataFrame(data_intersection, crs='EPSG:3857')
        

        return gdf_bbox, gdf_intersection
    

    def get_randomize_center_bbox(self, df_glacid, spatial_resolution = 30, img_res = 256):

        """
        Applies, when area of bbox is greater than area of glacier polygon

        Returns a random bbox around the glacier polygon. Outer bbox is moved around inner bbox randomly, so the generator isn't biased on centered geometries

        parameters:
        df_glacid : GeoDataFrame
            GeoDataFrame containing the glacier polygon
        spatial_resolution : int
            Spatial resolution of the image in meters
        img_res : int
            Resolution of the image in pixels
        """
        
        height = img_res * spatial_resolution
        width = img_res * spatial_resolution
        
        # Ensure the glacier polygon is in EPSG:3857, so meter base coordinates
        df_glacid = df_glacid.to_crs('EPSG:3857')

        glac_id = df_glacid['glac_id'].iloc[0]

        # get the glacier polygon
        glacier_polygon = df_glacid['geometry'].iloc[0]
        
        # get the bounds of the glacier polygon
        minx_s, miny_s, maxx_s, maxy_s = glacier_polygon.bounds

        # get center of glacier polygon
        center = df_glacid['geometry'].centroid.iloc[0]
        x, y = center.x, center.y

        # generate large bbox
        minx_l = x - width/2
        maxx_l = x + width/2
        miny_l = y - height/2
        maxy_l = y + height/2

        rand_center_x = random.uniform(minx_l + (maxx_s - minx_s), maxx_l - (maxx_s - minx_s))
        rand_center_y = random.uniform(miny_l + (maxy_s - miny_s), maxy_l - (maxy_s - miny_s))

        rand_minx = rand_center_x - width/2
        rand_maxx = rand_center_x + width/2
        rand_min_y = rand_center_y - height/2
        rand_maxy = rand_center_y + height/2    

        # create the random bbox
        rand_bbox = box(rand_minx, rand_min_y, rand_maxx, rand_maxy)

        # get intersection
        intersection = rand_bbox.intersection(glacier_polygon)

        # Create properties for the GeoDataFrame
        data_bbox = {'geometry': [rand_bbox], 'glac_id': [glac_id]}
        data_intersection = {'geometry': [intersection], 'glac_id': [glac_id]}

        # Create GeoDataFrames
        gdf_bbox = gpd.GeoDataFrame(data_bbox, crs='EPSG:3857')
        gdf_intersection = gpd.GeoDataFrame(data_intersection, crs='EPSG:3857')

        return gdf_bbox, gdf_intersection
    
    
    def rasterize_mask(self, df_glacid, spatial_resolution = 30, img_res = 256): # Probably not necessary
    
        """
        Function to rasterize the mask of the glacier polygon
    
        Parameters:
        df_glacid : GeoDataFrame
            GeoDataFrame containing the glacier polygon
    
        spatial_resolution : int
            The spatial resolution of the image in meters. Like 30m per pixel
    
        img_res : int
            The width and height of the image. Like 256 x 256 Pixels
        """

        height = img_res * spatial_resolution
        width = img_res * spatial_resolution
        # Ensure, that dataframe is in EPSG:3857
        df_glacid = df_glacid.to_crs('EPSG:3857')
    
        # get glac_id of glacier
        glac_id = df_glacid['glac_id'].iloc[0]
    
        # get the glacier polygon
        glacier_polygon = df_glacid['geometry'].iloc[0]
    
        # get the center of the glacier polygon
        center = df_glacid['geometry'].centroid.iloc[0]
    
        # Get x and y of center
        x, y = center.x, center.y
    
        # Define coordinates for Rectangle
        minx = x - width / 2 
        miny = y - height / 2 
        
        maxx = x + width / 2 
        maxy = y + height / 2 
    
        # make intersection mask
        intersection_mask = np.zeros((img_res, img_res), dtype=np.float32)
        intersection_mask = rasterize([(glacier_polygon, 1)], out_shape=(img_res, img_res), transform=rasterio.transform.from_bounds(minx, miny, maxx, maxy, img_res, img_res), dtype=np.float32)
        
        return intersection_mask
    
    def bbox_2_mask_rasterized(self, gdf_outer_bbox, gdf_inner_bbox, gdf_intersection, gdf_intersection_small, properties = {
        'outer_dim' : (256, 256),
        'inner_dim' : (64, 64),
    }):
    
        """
        Function to generate masks for the dataset. Rasterize seems to be more precise than geometry_mask
        Parameters:
        gdf_outer_bbox : GeoDataFrame
            GeoDataFrame containing the outer bbox. Essentially the bbox used to download the satellite image

        gdf_inner_bbox : GeoDataFrame
            GeoDataFrame containing the inner bbox. The inner bbox is generated, to later feed an input into the local discriminator.

        gdf_intersection : GeoDataFrame
            GeoDataFrame containing the intersection of the glacier polygon and the outer bbox. This is used to generate the mask for the whole glacier.

        gdf_intersection_small : GeoDataFrame
            GeoDataFrame containing the intersection of the glacier polygon and the inner bbox. This is used to generate the mask for the inner intersection.
            It should function as further information for the local discriminator.
        """
        
        # Get the difference between the outer bbox and the inner bbox
        difference = gdf_outer_bbox.difference(gdf_inner_bbox)

        # Make a mask for inner and outer bbox
        mask = np.zeros((properties['outer_dim']), dtype=np.float32)
        mask = rasterize(difference, out_shape=properties['outer_dim'], transform=rasterio.transform.from_bounds(*difference.total_bounds,
                                                                                                                properties['outer_dim'][0],
                                                                                                                properties['outer_dim'][1]),
                                                                                                                dtype=np.float32)
        # Invert the mask with numpy:
        inner_outer_mask = np.where(mask == 1, 0, 1) #TODO: Check if this is correct. Probably changes dtype to int instead of float
        inner_outer_mask = np.float32(inner_outer_mask)

        # Make the mask for the whole glacier
        mask = np.zeros((properties['outer_dim']), dtype=np.float32)
        intersection_mask = rasterize(gdf_intersection['geometry'], out_shape=properties['outer_dim'], transform=rasterio.transform.from_bounds(*gdf_outer_bbox['geometry'].total_bounds,
                                                                                                                properties['outer_dim'][0],
                                                                                                                properties['outer_dim'][1]),
                                                                                                                dtype=np.float32)
        
        # Make the mask for the inner intersection
        mask = np.zeros((properties['inner_dim']), dtype=np.float32)
        intersection_mask_small = rasterize(gdf_intersection_small['geometry'], out_shape=properties['inner_dim'], transform=rasterio.transform.from_bounds(*gdf_inner_bbox['geometry'].total_bounds,
                                                                                                                properties['inner_dim'][0],
                                                                                                                properties['inner_dim'][1]),
                                                                                                                dtype=np.float32)


        return inner_outer_mask, intersection_mask, intersection_mask_small
    
    def glacier_area_check(self, spatial_resolution, img_res, df_glacid):
        """
        Function to check if glacier bounds area is larger than mask area. If so, the function will return True, else False

        Parameters:
        spatial_resolution : int
            The spatial resolution of the image in meters. Like 30m per pixel

        img_res : int
            The width and height of the image. Like 256 x 256 Pixels

        """

        # Calculate the width and height of the image in meters (256 * 30)
        coord_res = spatial_resolution * img_res    

        # Calculate the area of the bbox (256 * 30) ^ 2
        area_bbox = coord_res ** 2 / 10**6 # in square kilometers 

        # Ensure that the dataframe is in EPSG:3857
        df_glacid = df_glacid.to_crs('EPSG:3857')
        #df_glacid = df_glacid.to_crs({'proj':'cea'})

        # Glacier bounds 
        minx, miny, maxx, maxy = df_glacid['geometry'].iloc[0].bounds
        
        # Make glacier bounds bbox
        bbox = box(minx, miny, maxx, maxy)

        # Calculate the area of the glacier bounds
        area_glacier_bounds = bbox.area / 10**6

        # Check if glacier bounds area is larger than mask area
        if area_glacier_bounds > area_bbox:
            return True
        else:
            return False
        
    def filter_glaciers_by_country(self, glacier_gdf, countries_to_exclude, buffer_distance=50000):
        """
        Filters glaciers in a GeoDataFrame based on whether their geometries intersect 
        with the polygons of countries to exclude. Buffer_distance is used to avoid edge cases. It is dependent on the CRS of the GeoDataFrame.

        Args:
            glacier_gdf (GeoDataFrame): A GeoDataFrame of glacier outlines.
            countries_to_exclude (list): A list of country names to exclude.

        Returns:
            GeoDataFrame: A filtered GeoDataFrame with glaciers outside excluded countries.
        """

        world_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world_df = world_df.to_crs(epsg=3857) # Make sure CRS is aligned
        glacier_gdf = glacier_gdf.to_crs(epsg=3857)

        exclude_mask = world_df['name'].isin(countries_to_exclude)
        excluded_countries_gdf = world_df[exclude_mask]

        buffered_countries_gdf = excluded_countries_gdf
        buffered_countries_gdf['geometry'] = buffered_countries_gdf['geometry'].buffer(buffer_distance) # Buffer to avoid edge cases
        buffered_countries_gdf.reset_index(drop=True, inplace=True)

        filtered_glaciers = []
        for idx, row in glacier_gdf.iterrows():
            glacier_geometry = row['geometry']
            if not any(buffered_countries_gdf.intersects(glacier_geometry)):
                filtered_glaciers.append(row)

        return gpd.GeoDataFrame(filtered_glaciers, crs=glacier_gdf.crs)   
    
    def check_sum_inner_outer(self, inner_outer, dim = (64,64)):

        """Checks if the sum of the inner-outer mask is equal to 4096"""

        value = dim[0] * dim[1]

        if np.sum(inner_outer) != value: # Checks, if the sum of the inner_outer mask is appropriate, if not, return False
            return False
        else:
            return True
        
    def glacier_fits_bbox_forDataframe(self, row, spatial_resolution, img_res):
        """
        Checks if a glacier outline (represented by a row in a DataFrame)
        fits within a bounding box defined by the spatial resolution and image resolution.

        Parameters:
        row : pd.Series
            A row from a DataFrame containing a 'geometry' column with the glacier outline.
        spatial_resolution : int
            The spatial resolution of the image in meters (e.g., 30 meters per pixel).
        img_res : int
            The width and height of the image (e.g., 256 pixels).

        Returns:
        bool
            True if the glacier fits within the bbox, False otherwise.
        """

        coord_res = spatial_resolution * img_res
        area_bbox = coord_res ** 2 / 1e6  # Area of the bounding box in sq. km

        geometry = row
        minx, miny, maxx, maxy = geometry.bounds
        bounds_glacier = box(minx, miny, maxx, maxy)

        area_glacier = bounds_glacier.area / 1e6

        return area_glacier <= area_bbox

    
              
                
    def make_mask_from_file(self, df, path_input = None, path_output = None): 

        """
        Generates negative masks for the training. Takes a DataFrame with glacier outlines as well as corresponding DEMs as input. 
        df: DataFrame with glacier outlines
        path_input: Path to the DEMs
        path_output: Path to save the masks

        # Probably deprecated. Has not been utilized yet
        """

        if path_input is None or path_output is None:
            raise ValueError("Path required")
        
        for r, d, f in os.walk(path_input):
            for file in f:
                filepath = os.path.join(r, file)
                glac_id = file[:-4]
                df_temp = df[df['glac_id'] == glac_id]
                
                
                with rasterio.open(filepath) as src:
                    dem_bounds = src.bounds
                    transform = src.transform
                    out_shape = src.shape
                    profile = src.profile

                # create an empty mask 
                mask = np.zeros(out_shape, dtype=np.float32)
                geom = shape(df_temp['geometry'].values[0])
                mask = rasterize(shapes = [geom], out_shape = out_shape, transform = transform)
                mask = np.where(mask == 0, 1, 0)

                profile.update({'driver': 'GTiff', 
                    'height': mask.shape[0],
                    'width': mask.shape[1],
                    'dtype': 'float32',  # Make sure the data type is correct for the mask
                    'transform': transform})
                
                out_name = os.path.join(path_output, f"mask_{glac_id}.tif")
                
                with rasterio.open(out_name, 'w', **profile) as dst:
                    dst.write(mask, 1) # mask and number of bands
 


class DataLoader:

    def load_demLEGACY(self, path,shape = (256,256)):
        # some_list = []
        for root, dirs, files in os.walk(path):
            for file in files:

                f_path = os.path.join(root,file)

                # Check if file exists:
                if not os.path.isfile(f_path):
                    print(f"File {f_path} does not exist")
                    continue
                
                try:
                    data = cv2.imread(f_path.decode('UTF-8'), cv2.IMREAD_LOAD_GDAL)
                    data = cv2.resize(data, dsize=shape, interpolation=cv2.INTER_AREA)

                    image = tf.convert_to_tensor(data)
                    image = tf.expand_dims(image, axis = -1) # Necessary for image crop. Either here or in crop function
                    yield image

                    #some_list.append(image)
                except Exception as e:
                    print(f"Error while reading {f_path}: {e}")

        
        # return some_list

    def load_maskLEGACY(self, path, shape = (256,256)):
        for root, dirs, files in os.walk(path):
            for file in files:
                f_path = os.path.join(root,file)
                #print('File Path to be checked: ', f_path)

                data = np.load(f_path)
                data_tensor = tf.convert_to_tensor(data)
                data_tensor = tf.expand_dims(data_tensor, axis = -1)
                mask = tf.image.resize(data_tensor, shape, method = 'nearest')
                yield mask

                #some_list.append(data_tensor)

        #return some_list
        
    def load_dem_cv2(self, file_path, resize = False, shape = (256, 256)):

        """
        Utilize cv2 to load dem for training.
        """

        for file in file_path:
            try:
                data = cv2.imread(file.decode('UTF-8'), cv2.IMREAD_LOAD_GDAL)
                if resize:
                    data = cv2.resize(data, dsize=shape, interpolation=cv2.INTER_AREA)

                image = tf.convert_to_tensor(data)
                image = tf.expand_dims(image, axis=-1)

                yield image
            except Exception as e:
                print(f'Error: {e}')
                raise ValueError(f'Error in load_dem_cv2: {e}')

    def load_mask(self, file_path, shape = (256, 256)):
            """
            Loading masks for training
            """
        
            for file in file_path:
                data = np.load(file.decode('UTF-8'))
                data_tensor = tf.convert_to_tensor(data)
                data_tensor = tf.expand_dims(data_tensor, axis=-1)
                #data_tensor = tf.image.resize(data_tensor, size=shape, method='nearest')

                yield data_tensor

    def populate_list(self, file_path):
        """
        Returns a list of files in a folder
        """

        f_names = []

        for root, dirs, files in os.walk(file_path):
            for file in files:
                f_path = os.path.join(root, file)
                
                f_names.append(f_path)
        f_names.sort()

        return f_names
    
    def shuffle_lists(self, lists, seed=None):
        """
        Shuffle multiple lists in the same order
        """
        if seed is not None:
            random.seed(seed)  
        zipped = list(zip(*lists))
        random.shuffle(zipped)
        return zip(*zipped)
    

class DataCleaner:

    def change_mask_dtype(self, file_path, output_path):
        """
        Changes the data type of the masks in a folder and saves them in a new folder.

        # Probably deprecated. Corrected error in other function
        """
        
        for file in os.listdir(file_path):
            mask = np.load(os.path.join(file_path, file))
            mask = mask.astype(np.float32)
            np.save(os.path.join(output_path,file), mask)
    

class DaskUtils:

    """
    Only to load a large shapefile using dask-geopandas. Further processing of the Datafiles will be handled by GISProcessor.
    """

    def __init__(self,npartitions = 6):
        self.npartitions = npartitions
        pass


    def filter_shapefile(self, filepath, target_columns=None, use_std_col = 'yes', dt_col = None,
                         dt_format = None): #USED IN SCRIPT
        """
        Loads a large shapefile using dask-geopandas, displays available columns,
        allows the user to select desired columns, and returns a filtered DataFrame.

        Args:
            filepath (str): Path to the shapefile.
            target_columns (list, optional): A list of pre-selected target columns. 
                                            If None, the user will be prompted to select columns.

        Returns:
            dask_geopandas.GeoDataFrame: The filtered GeoDataFrame.

        Example Usage:
        
        utils = DaskUtils(npartitions=6)

        Define target columns
        target_columns = ['line_type', 'glac_id', 'area', 'src_date', 'glac_name', 'glac_stat']

        Read in the shapefile
        ddf = utils.filter_shapefile(filepath=os.path.join(fp_desktop, "glims_polygons.shp"),
                                    target_columns=target_columns, dt_col = 'src_date', 
                                    dt_format = '%Y-%m-%dT%H:%M:%S')

        ddf_bound = ddf[ddf['line_type'] == 'glac_bound'] # Filter the dataframe for glacier boundaries 
        ddf_intrnl = ddf[ddf['line_type'] == 'intrnl_rock'] # Filter the dataframe for internal rock

        Use Dask only to load and pre select the data. Everything else is to complicated and done with pandas:
        """

        ddf = dask_gpd.read_file(filepath, npartitions=self.npartitions)

        if use_std_col == 'yes':
            print('Using standard columns:\n')
            # Use standard columns
            target_columns = ['line_type', 'glac_id', 'area', 'src_date', 'glac_name', 'glac_stat']
            print(target_columns)

        # Get user input if target columns are not provided 
        if target_columns is None:
            # Display available columns
            print("Available columns:\n", ddf.columns.to_list())
            while True:
                target_columns = input("Enter the desired columns separated by commas (q to quit): ").split(',')
                if target_columns == ['q']: # Quit
                    print("Operation cancelled.")
                    return  None
                elif all(col in ddf.columns for col in target_columns):
                    break
                else:
                    print("Invalid column(s). Please try again.")

    
        target_columns.extend(['geometry', 'src_date','glac_id'])  # Always keep certain columns
        target_columns = list(dict.fromkeys(target_columns)) # Remove duplicates

        # Filter the DataFrame
        ddf_filtered = ddf[target_columns]  

        # Convert the date column to datetime
        """
        if dt_col is not None:
            ddf_filtered[dt_col] = pd.to_datetime(ddf_filtered[dt_col], format='%Y-%m-%dT%H:%M:%S')
        else:
            print("No date column provided. The date column will not be converted to datetime.")
        
        """
        if dt_col is not None:
            if dt_format is not None:
                # typically '%Y-%m-%dT%H:%M:%S'
                ddf_filtered[dt_col] = ddf_filtered[dt_col].map_partitions(
                    pd.to_datetime, format = dt_format, meta = 
                    (dt_col, 'datetime64[ns]'))
            else:
                print('No date format provided. The date column will not be converted to datetime.')
        else:
            print("No date column provided. The date column will not be converted to datetime.")

        return ddf_filtered


class DataIntegrity:


    def check_sum_inner_outer(self, inner_outer_path, dim = (64,64)):

        """Checks if the sum of the inner-outer mask is equal to dim * dim"""
        sum_value = dim[0] * dim[1]
        corrupt_files = []
        for filename in tqdm(os.listdir(inner_outer_path)):
            filepath = os.path.join(inner_outer_path, filename)
            data = np.load(filepath)
            if np.sum(data) != sum_value:
                corrupt_files.append(os.path.join(inner_outer_path, filename).encode())
        return corrupt_files



    def check_min_max_values(self, folder_path, shape = (256,256)):

        """Checks if the min and max values of the DEMs are corrupt, to avoid division by zero errors"""

        corrupt_files = []

        for file in tqdm(os.listdir(folder_path)):
            filepath = os.path.join(folder_path, file)
            data = cv2.imread(filepath, cv2.IMREAD_LOAD_GDAL)
            data = cv2.resize(data, dsize=shape, interpolation=cv2.INTER_AREA)
            if np.max(data) - np.min(data) == 0:
                corrupt_files.append(os.path.join(folder_path, file).encode())
        return corrupt_files




    def resize_dems(self, dem_path, shape = (256,256)):

        """Resizes the DEMs to the desired shape"""

        for file in tqdm(os.listdir(dem_path)):
            filepath = os.path.join(dem_path, file)
            data = cv2.imread(filepath, cv2.IMREAD_LOAD_GDAL)
            data = cv2.resize(data, dsize=shape, interpolation=cv2.INTER_AREA)
            cv2.imwrite(filepath, data)




    def delete_corrupt_files(self, corrupt_files, dem_path, inner_outer_path, intersection_path ,intersection_small_path, base_path):

        """Deletes corrupt files from the dataset
        Args:
            corrupt_files (list): List of corrupt files
            dem_path (str): Path to DEM files
            inner_outer_path (str): Path to Inner-Outer Mask files
            intersection_path (str): Path to Intersection Mask files
            intersection_small_path (str): Path to Intersection Mask small files"""

        for file in corrupt_files:
            
            if file.decode('UTF-8').endswith('.tif'):
                dem_file = file.decode('UTF-8')
            
            if file.decode('UTF-8').endswith('.npy'):
                dem_file = os.path.join(dem_path, os.path.basename(file.decode('UTF-8').replace('.npy', '.tif')))
                                        
            inner_outer_file = os.path.join(inner_outer_path, os.path.basename(file.decode('UTF-8')).replace('.tif', '.npy'))
            intersection_file = os.path.join(intersection_path, os.path.basename(file.decode('UTF-8')).replace('.tif', '.npy'))
            intersection_small_file = os.path.join(intersection_small_path, os.path.basename(file.decode('UTF-8')).replace('.tif', '.npy'))

            os.remove(dem_file)
            os.remove(inner_outer_file)
            os.remove(intersection_file)
            os.remove(intersection_small_file)

            with open(os.path.join(base_path, 'corrupt_files.txt'), 'a') as f:
                for item in corrupt_files:
                    f.write("%s\n" % item)

        print(f'Deleted {len(corrupt_files)} corrupt files')

    def norm_dems(self, dem_path):

        """Normalizes the DEMs to the range [-1, 1]. This is not utilized in the current script
           Instead, values are normalized during data loading. This function is kept for reference.
        """

        for file in tqdm(os.listdir(dem_path)):
            filepath = os.path.join(dem_path, file)
            data = cv2.imread(filepath, cv2.IMREAD_LOAD_GDAL)
            min_val = np.min(data)
            max_val = np.max(data)

            if max_val - min_val == 0:
                raise ValueError(f'Min and max values are equal for file {file}')
            
            data = (data - min_val) / (max_val - min_val) * 2 -1 # normalize to [-1,1]
            cv2.imwrite(filepath, data)


