# How to Create Transect and Reference Shoreline Files GeoJSON in QGIS
## 1. QGIS
QGIS is an opensource geospatial information system software. It can be downloaded <a href='https://qgis.org/'>here</a>. QGIS can be used to create both the transects and reference shoreline GeoJSON saved in `user_inputs/<region>/<sitename>` as `<sitename>_transects.geojson` and `<sitename>_shoreline.geojson` respectively.
## 2. Transect GeoJSON
### 2.1 Create Layer
First, Drag and drop a tiff file (downloaded by the API here: `data/<region>/<sitename>/*_3B_AnalyticMS_toar_clip.tif`) of your choosing into QGIS.

Next, click the "create shapefile layer" botton shown below.

<img src="media\01_select_create_layer.JPG" alt="create shapefile layer" style="max-width:70%">

Once clicked the dialog below will appear. Select LineString for geometry type and write filename (the filename will not be the final name so it is arbitraty).

<img src="media\02_create_transects_linestring_dialog.JPG" alt="create shapefile layer dialog" style="max-width:70%">

### 2.2 Draw Transects
Select the penicil button and then the digitizing tool bar button the the right of it to begin drawing.
<div style="display: flex; align-items: flex-start;">
    <div style="margin-right: 40px;">
        <img src="media\03_select_pencil.JPG" alt="select pencil" style="max-width: 100%; height: 100%;">
    </div>
    <div>
        <img src="media\04_select_digitizing_toolbar.JPG" alt="select digitizing toolbar" style="max-width: 100%; height: 100%;">
    </div>
</div>

For each transect right click at the desired *landward* end of the transect then again at the seaward end. Then left click to complete the transect.

<img src="media\02_create_transects_linestring_dialog.JPG" alt="create shapefile layer dialog" style="max-width:70%">

### 2.3 Export File


