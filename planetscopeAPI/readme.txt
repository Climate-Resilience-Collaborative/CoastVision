This directory contains code to use the PlanetScope data and orders API.
The data API is best for searching items (a satellite observation given a time and area).
Because, it takes a while for the API to respond to a download request it is inefficient for bulk downloads.
However, the orders API is made for bulk "orders" but we need the the item ids for the items we would like to download (from the data api).
So, we use the data API to search item ids based on varius filters (timeframe, AOI) then we pass that list of ids to the orders API for download.

Joel Nicolow, Climate Resilience Collaborative, School of Ocean and Earth Science and Technology, University of Hawaii at Manoa (April 21, 2022)