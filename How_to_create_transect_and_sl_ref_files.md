# How to Create Transect and Reference Shoreline Files GeoJSON in QGIS
## 1. QGIS
QGIS is an opensource geospatial information system software. It can be downloaded <a href='https://qgis.org/'>here</a>. QGIS can be used to create both the transects and reference shoreline GeoJSON saved in `user_inputs/<region>/<sitename>` as `<sitename>_transects.geojson` and `<sitename>_shoreline.geojson` respectively.
## 2. Transect GeoJSON
### 2.1 Create Layer
First, Drag and drop a tiff file (downloaded by the API here: `data/<region>/<sitename>/*_3B_AnalyticMS_toar_clip.tif`) of your choosing into QGIS.

Next, click the "create shapefile layer" botton shown below.

<img src="media\01_select_create_layer.JPG" alt="create shapefile layer" style="max-width:50%">

Once clicked the dialog below will appear. Select LineString for geometry type and write filename (the filename will not be the final name so it is arbitraty).

<img src="media\02_create_transects_linestring_dialog.JPG" alt="create shapefile layer dialog" style="max-width:50%">

### 2.2 Draw Transects
Select the penicil button and then the digitizing tool bar button the the right of it to begin drawing.
<img src="media\03_select_pencil.JPG" alt="select pencil" style="max-width: 50%; height: 100%;">
<img src="media\04_select_digitizing_toolbar.JPG" alt="select digitizing toolbar" style="max-width: 50%; height: 100%;">

For each transect right click at the desired *landward* end of the transect then again at the seaward end. Then left click to complete the transect.

<img src="media\02_create_transects_linestring_dialog.JPG" alt="create shapefile layer dialog" style="max-width:50%">

### 2.3 Export File

Once all transects have been drawn left click on the layer on the left hand menu. Then select "Export" and then "Save Features As..."
<img src="media\06_expot_menu.png" alt="export menu" style="max-width:50%">

After selecting this save a save file popup will open. Save the file `user_vinputs/<region>/<sitename>/<sitename>_transects.geojson`.
<img src="media\07_save_file_dialog.JPG" alt="export menu" style="max-width:50%">

:warning: When Choosing CRS choose the projection that the TIFF image you are using is in. See below.
<img src="media\07.5_save as project.JPG" alt="CRS" style="max-width:50%">

## 3. Reference Shoreline
For the reference shoreline the steps are the same except instead of making multimple line segments (transects) just create one linestring that is the entire reference shoreline and save it as `user_vinputs/<region>/<sitename>/<sitename>_shoreline.geojson`.

<img src="media\08_save_shoreline.JPG" alt="save sl ref" style="max-width:50%">




