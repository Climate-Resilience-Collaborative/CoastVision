"""
This script contains functions to setup a CoastVision site given notebook inputs

Author: Joel Nicolow, Climate Resiliance Collaborative, School of Ocean and Earth Science and Technology (September, 2024)
"""
import os
import json

def create_site_dict_json_for_API(site_name:str, aoi:list, start_date:str, end_date:str):
    """
    This function creates sites/<sitename>_site_dict.json for the given site
    """

    if not os.path.exists('sites'): os.mkdir('sites')

    site_dict = {f"{site_name}": {"item_type": "PSScene", "geometry_filter": {"type": "GeometryFilter", "field_name": "geometry", "config": {
        "type": "Polygon", 
        "coordinates": [aoi]}},
        "date_range_filter": {"type": "DateRangeFilter", "field_name": "acquired", 
                              "config": {"gte": f"{start_date}T00:00:00.000Z", "lte": f"{end_date}T00:00:00.000Z"}}, "cloud_cover_filter": {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": 0.3}}}}

    with open(os.path.join('sites', f'{site_name}_site_dict.json'), "w") as f: 
        json.dump(site_dict, f, indent=4) # indent=4 makes the JSON pretty-printed

    return site_dict


def write_api_key_file(api_key:str):
    if not os.path.exists('sites'): os.mkdir('sites')
    
    file_path = os.path.join('sites', 'PlanetScope_API_key.txt')
    with open(file_path, "w") as file:
        file.write(api_key)