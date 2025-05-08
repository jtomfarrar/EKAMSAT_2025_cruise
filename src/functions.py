## THREDDS tools
# Adapted from a script was obtained from an OOI webpage:
# https://oceanobservatories.org/knowledgebase/how-can-i-download-all-files-at-once-from-a-data-request/
# Written by Sage 4/5/2016, revised 5/31/2018

# url for the data (use xml extension instead of html)
# Everything before "catalog/"
# server_url = 'http://smode.whoi.edu:8080/thredds/'
# Everything from "catalog/" on:
# request_url = 'catalog/IOP2_2023/satellite/VIIRS_NPP/catalog.xml'
# url = server_url + request_url

import simplekml

def get_elements(url, tag_name, attribute_name):
    """Get elements from an XML file"""
    from xml.dom import minidom
    from urllib.request import urlopen
    from urllib.request import urlretrieve

    # usock = urllib2.urlopen(url)
    usock = urlopen(url)
    xmldoc = minidom.parse(usock)
    usock.close()
    tags = xmldoc.getElementsByTagName(tag_name)
    attributes=[]
    for tag in tags:
        attribute = tag.getAttribute(attribute_name)
        attributes.append(attribute)
    return attributes
 
def download_THREDDS(server_url, request_url):
    '''Download all the netcdf files from a given THREDDS catalog.

    TODO: It would be nice to specify the directory where the files should be written.

    Parameters
    ----------
    server_url : str
        Everything before "catalog/" in THREDDS catalog address, e.g.,
        server_url = 'http://smode.whoi.edu:8080/thredds/'
    request_url : str
        Everything after (and including) "catalog/"

    Returns
    -------
    None

    Example
    -------
    URL for the data (use xml extension instead of html)
    For 'http://smode.whoi.edu:8080/thredds/catalog/IOP2_2023/satellite/VIIRS_NPP/catalog.xml',
    Everything before "catalog/"
    server_url = 'http://smode.whoi.edu:8080/thredds/'
    Everything from "catalog/" on:
    request_url = 'catalog/IOP2_2023/satellite/VIIRS_NPP/catalog.xml'
    '''
    url = server_url + request_url
    print(url)
    catalog = get_elements(url,'dataset','urlPath')
    files=[]
    for citem in catalog:
        if (citem[-3:]=='.nc'):
            files.append(citem)
    count = 0
    for f in files:
        count +=1
        file_url = server_url + 'fileServer/' + f
        file_prefix = file_url.split('/')[-1][:-3]
        file_name = file_prefix  + '.nc'
        print('Downloaing file %d of %d' % (count,len(files)))
        print(file_url)
        a = urlretrieve(file_url,file_name)
        print(a)

def list_THREDDS(server_url, request_url):
    '''Return a list of OPeNDAP links for all the netcdf files from a given THREDDS catalog.

    Parameters
    ----------
    server_url : str
        Everything before "catalog/" in THREDDS catalog address, e.g.,
        server_url = 'http://smode.whoi.edu:8080/thredds/'
    request_url : str
        Everything after (and including) "catalog/"

    Returns
    -------
    file_list : list of strings
        list of OPeNDAP/DODS links that can be used to access the data

    Example
    -------
    URL for the data (use xml extension instead of html)
    For 'http://smode.whoi.edu:8080/thredds/catalog/IOP2_2023/satellite/VIIRS_NPP/catalog.xml',
    Everything before "catalog/"
    server_url = 'http://smode.whoi.edu:8080/thredds/'
    Everything from "catalog/" on:
    request_url = 'catalog/IOP2_2023/satellite/VIIRS_NPP/catalog.xml'
    '''
    url = server_url + request_url
    file_list=[]
    print(url)
    catalog = get_elements(url,'dataset','urlPath')
    files=[]
    for citem in catalog:
        if (citem[-3:]=='.nc'):
            files.append(citem)
    count = 0
    for f in files:
        file_url = server_url + 'dodsC/' + f
        file_prefix = file_url.split('/')[-1][:-3]
        file_name = file_prefix  + '.nc'
        file_list.append(file_url)
        print(file_url)
    return file_list

# %%
def create_kml_file(kml_name, plot_file, overlay_name, pts_lon, pts_lat, BD, mooring, xmin, xmax, ymin, ymax):
    """
    Create a KML file with SST data and mooring points.

    Parameters:
    - kml_name: Name of the KML file to save.
    - plot_file: File path of the saved plot image.
    - xmin, xmax, ymin, ymax: Bounding box coordinates for the SST map.
    - pts_lon, pts_lat: Lists of longitudes and latitudes for BD/RAMA moorings.
    - mooring: Dictionary containing 'lon' and 'lat' for the 2024 site.
    - overlay_name: Name for the ground overlay in the KML file.
    """

    # Create a KML object
    kml = simplekml.Kml()

    # Add SST data as a ground overlay
    sst_kml = kml.newgroundoverlay(name=overlay_name)
    sst_kml.icon.href = plot_file  # Use the saved plot image
    sst_kml.latlonbox.north = ymax
    sst_kml.latlonbox.south = ymin
    sst_kml.latlonbox.east = xmax
    sst_kml.latlonbox.west = xmin

    # Add BD/RAMA moorings as points
    index = 0
    for lon, lat in zip(pts_lon, pts_lat):
        pnt = kml.newpoint(name=BD[index], coords=[(lon, lat)])
        pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        pnt.style.iconstyle.color = simplekml.Color.magenta
        index += 1

    # Add the 2024 site as a point
    for lon, lat in zip(mooring['lon'], mooring['lat']):
        pnt = kml.newpoint(name="2024 Site", coords=[(lon, lat)])
        pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        pnt.style.iconstyle.color = simplekml.Color.black

    # Save the KML file
    kml.save(kml_name + '.kml')
    print(f"KML file '{kml_name}.kml' created successfully.")


def waypoints_to_kml(kml_name, wpt):
    """
    Create a KML file with waypoints.

    Parameters:
    - kml_name: Name of the KML file to save.
    - wpt: Dictionary containing 'lon' and 'lat' for the waypoints.
    """
    
    # Create a KML object
    kml = simplekml.Kml()

    # Add waypoints as points
    for lon, lat in zip(wpt['lon'], wpt['lat']):
        pnt = kml.newpoint(name="Waypoint", coords=[(lon, lat)])
        pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        pnt.style.iconstyle.color = simplekml.Color.red
        pnt.style.iconstyle.scale = 1.5

    # Save the KML file
    kml.save(kml_name + '.kml')
    print(f"KML file '{kml_name}.kml' created successfully.")

def points_to_kml(kml_name, wpt, pt_label):
    """
    Create a KML file with waypoints.

    Parameters:
    - kml_name: Name of the KML file to save.
    - wpt: Dictionary containing 'lon' and 'lat' for the waypoints.
    """
    
    # Create a KML object
    kml = simplekml.Kml()

    # Add waypoints as points
    for lon, lat in zip(wpt['lon'], wpt['lat']):
        pnt = kml.newpoint(name=pt_label, coords=[(lon, lat)])
        pnt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        pnt.style.iconstyle.color = simplekml.Color.green

    # Save the KML file
    kml.save(kml_name + '.kml')
    print(f"KML file '{kml_name}.kml' created successfully.")
