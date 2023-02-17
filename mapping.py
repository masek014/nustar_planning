from __future__ import annotations
import datetime
import sunpy.map
import astropy.units as u
from astropy.units import Quantity
import matplotlib.pyplot as plt
import matplotlib.patches as patches # for rectangles in Sunpy V3.1.0, I can't get draw_rectangle() to work
import numpy as np

from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import Helioprojective, RotatedSunFrame, transform_with_sun_center
from sunpy.net import Fido, attrs as a

import warnings
warnings.filterwarnings("ignore")


DT_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Define the universal axes limits to be used by all plots.
# In units of arcseconds.
X_MIN = -1200
Y_MIN = -1200
X_MAX = 1200
Y_MAX = 1200


def most_recent_map(_map: str, wavelength: Quantity):
    """
    Query and return the most recent _map (of "AIA" or "STEREO") 
    that Sunpy.Fido can get.
    
    Parameters
    ----------
    _map : string
        The instrument you want the most recent map from. 
        E.g., _map="aia" or _map="stereo". 
        
    Returns
    -------
    Sunpy generic map.
    """
    
    if _map.lower() == 'aia':
        info = (a.Instrument('aia') & a.Wavelength(wavelength))
        look_back = {'minutes': 30}
    elif _map.lower() == 'stereo':
        info = (a.Source('STEREO_A') & a.Instrument('EUVI') & \
                a.Wavelength(wavelength)) 
        look_back = {'days': 5}
    else:
        print("Don't know what to do! Please set _map=\"aia\" or \"stereo\".")
        
    current = datetime.datetime.now()
    current_date = current.strftime(DT_TIME_FORMAT)

    past = current-datetime.timedelta(**look_back)
    past_date = past.strftime(DT_TIME_FORMAT)
    startt = str(past_date)
    endt = str(current_date)

    result = Fido.search(a.Time(startt, endt), info)
    file_download = Fido.fetch(result[0, -1], site='ROB')

    data_map = sunpy.map.Map(file_download[-1])
    bl = SkyCoord(X_MIN*u.arcsec, Y_MIN*u.arcsec, frame=data_map.coordinate_frame)
    tr = SkyCoord(X_MAX*u.arcsec, Y_MAX*u.arcsec, frame=data_map.coordinate_frame)
    data_map = data_map.submap(bottom_left=bl, top_right=tr)
    
    return data_map 


def plot_map(in_map: sunpy.map, ax: plt.Axes, cmap: str = 'gray', title: str = ''):
    """
    Plot the provided map on the provided axes.

    Parameters
    ----------
    in_map : Sunpy map
        The input map to be projected.
    ax : matplotlib.pyplot axes object
        The axes the NuSTAR fov is to be drawn on.
    cmap : str
        The matplotlib colormap to use on the scale.
    title : str
        The title of the plot.
    """
    
    in_map.plot(cmap=cmap)
    in_map.draw_limb(color='black', zorder=1)
    ax.tick_params(which='major', direction='in')
    ax.grid(False)
    ax.set(xlabel='X (arcseconds)', ylabel='Y (arcseconds)', title=title)
    (ax.coords[0]).display_minor_ticks(True)
    (ax.coords[0]).set_minor_frequency(5)
    (ax.coords[1]).display_minor_ticks(True)
    (ax.coords[1]).set_minor_frequency(5)
    ax.tick_params(which='minor', length=1.5)
    # plt.colorbar(ax=ax, pad=0.01, shrink=0.85, fraction=0.06, aspect=50)
    

def project_map(in_map, future_time):
    """
    Create a projection of an input map at the given input time.
    
    Parameters
    ----------
    in_map : Sunpy map
        The input map to be projected.
    future_time : str
        The time of the projected map. 
        
    Returns
    -------
    Sunpy generic map.
    """

    in_time = in_map.date
    out_frame = Helioprojective(observer='earth', obstime=future_time,
                                rsun=in_map.coordinate_frame.rsun)
    rot_frame = RotatedSunFrame(base=out_frame, rotated_time=in_time)

    out_shape = in_map.data.shape
    out_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=out_frame)
    header = sunpy.map.make_fitswcs_header(out_shape,
                                           out_center,
                                           scale=u.Quantity(in_map.scale))
    
    out_wcs = WCS(header)
    out_wcs.coordinate_frame = rot_frame

    with transform_with_sun_center():
        arr, _ = reproject_interp(in_map, out_wcs, out_shape)
    
    out_warp = sunpy.map.Map(arr, out_wcs)
    out_warp.plot_settings = in_map.plot_settings

    # Set the axes limits.
    bl = SkyCoord(X_MIN*u.arcsec, Y_MIN*u.arcsec, frame=out_warp.coordinate_frame)
    tr = SkyCoord(X_MAX*u.arcsec, Y_MAX*u.arcsec, frame=out_warp.coordinate_frame)
    out_warp = out_warp.submap(bottom_left=bl, top_right=tr)

    return out_warp


def draw_nustar_fov(in_map, ax, center, angle=0*u.arcsecond,
    layers=(-100, 0, 100)*u.arcsecond, colors='red', pixscale=None,
    b_mark_center=True):
    """
    Draw squares representing NuSTAR's field of view on the current map.
    
    By default, three squares are drawn: one that is equal
    to the 12x12 arcminute FOV and two with side lengths
    of +-100 arcseconds from the actual side lengths.
    
    Parameters
    ----------
    in_map : Sunpy map
        The input map on which the squares will be overlaid.
    ax : matplotlib.pyplot axes object
        The axes the NuSTAR fov is to be drawn on.
    center_x : float
        The x position, in arcseconds, of the squares' center point.
    center_y : float
        The y position, in arcseconds, of the squares' center point.
    layers : list
        List of values, in arcseconds, containing the adjustments
        to the side lengths of the drawn squares. Each value results
        in a new square drawn on the map.
    colors : str or list of str
        The colors of the drawn squares. If colors is a string,
        then each square will be drawn with that color.
        Otherwise, a list can be provided to customize the color
        of each layer. The index of the color will match the index
        of the layer.
    angle : int or float
        Anti-clockwise rotation from Solar north for NuSTAR field of view.
    pixscale : float
        Arcsecond-to-pixel conversion for the original AIA or STEREO map.
        Needed for the NuSTAR field of view rotation.
    b_mark_center : bool
        Specify whether the center of the fov should be marked.
        
    Returns
    -------
    None
    """
    
    # Change the colors variable to a list if it's not already one.
    if not isinstance(colors, list):
        colors = [colors]*len(layers)
    
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    FOV_SIDE_LENGTH = 720*u.arcsecond # 12'
    center_x, center_y = center.to(u.arcsecond)

    angle_str = ''
    for i, diff in enumerate(layers):
        linestyle = 'solid'
        if diff != 0:
            linestyle = 'dashed'
        if angle == 0:
            # Translate to bottom left corner of rectangle
            bottom_left = ((center_x - FOV_SIDE_LENGTH/2 - diff),
                           (center_y - FOV_SIDE_LENGTH/2 - diff))
            rect_bl = SkyCoord(*bottom_left, frame=in_map.coordinate_frame)

            in_map.draw_quadrangle(bottom_left=rect_bl,
                                   width=(FOV_SIDE_LENGTH+2*diff),
                                   height=(FOV_SIDE_LENGTH+2*diff),
                                   color=colors[i],
                                   linestyle=linestyle,
                                   axes=ax)
        else:
            # **kwargs dont get passed to matplotlib so not easy way to rotate, do it myself
            # get boxes in pixels, newer Sunpy doesn't allow draw_rectangle here
            
            # get bottom left coords in ref. frame where center of box 
            # bx_arc_square, by_arc_square = (-FOV_SIDE_LENGTH/2 - diff).value, (-FOV_SIDE_LENGTH/2 - diff).value
            bx_arc_square, by_arc_square = -FOV_SIDE_LENGTH/2 - diff, -FOV_SIDE_LENGTH/2 - diff

            # rotate bottom left clockwise to find where box to-be-rotated to maintain centers needs to be
            rot_mat = np.array([[np.cos(-angle.to(u.rad).value), np.sin(-angle.to(u.rad).value)],
                                [-np.sin(-angle.to(u.rad).value), np.cos(-angle.to(u.rad).value)]]) @ np.array([bx_arc_square.value,by_arc_square.value])

            bx_rotarc, by_rotarc = (rot_mat[0], rot_mat[1])*u.arcsecond
            
            # bottom left of new box in arcsec where center of the Sun is (0,0)
            bx_arc, by_arc = center_x+bx_rotarc, center_y+by_rotarc
            
            center_pix = (in_map.data.shape[1]/2, in_map.data.shape[0]/2)*u.pix
            rect_x = center_pix[0] + bx_arc/pixscale
            rect_y = center_pix[1] + by_arc/pixscale
            side_length = ((FOV_SIDE_LENGTH+2*diff)/pixscale)

            # create and rotate box
            rect = patches.Rectangle(
                [rect_x.value, rect_y.value],
                side_length.value,
                side_length.value,
                angle=angle.to(u.deg).value,
                facecolor='none',
                linewidth=1,
                linestyle=linestyle,
                edgecolor=colors[i]
            )
            ax.add_patch(rect)
        
        angle_str = f'Rotated {angle} Anti-clockwise'

        # if b_mark_center:
        #     circle = patches.Circle(
        #         [rect_x.value, rect_y.value],
        #         radius=5,
        #         color='black',
        #         facecolor='black'
        #     )
        #     ax.add_patch(circle)

        # Reset the axes since they may be slightly changed from the patch.
        ax.set_xlim(ax_xlim)
        ax.set_ylim(ax_ylim)

        # Add text on plot.
        # Determine the position of the text box.
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        x_mid, y_mid = (X_MAX+X_MIN)/2, (Y_MAX+Y_MIN)/2
        # text_x = ax_xlim[1] * (1-(x_mid-X_MIN)/(X_MAX-X_MIN))
        # if the fov centre is in the lower half then put text in the top and vice versa
        # text_y = ax_ylim[1] * (y_mid-Y_MIN-1050)/(Y_MAX-Y_MIN) if center_y>=0 else ax_ylim[1] * (1-(y_mid-Y_MIN-1050)/(Y_MAX-Y_MIN))
        
        text_x = 0.5 # 0.265 # Offset to the left
        text_y = 0.08 if center_y >= 0 else 0.92

        # To make the text dynamic, we need to format
        # the text string based on the layers list.
        layers_copy = [l.value for l in layers]
        if 0 in layers_copy:
            layers_copy.remove(0)

        # Convert the list of int to list of str, add arcsecond unit symbols,
        # and add '+' to positive numbers.
        layers_str = ''.join(f'+{x}\"' if x>0 else f'{x}\"' for x in layers_copy)
        text_str = f'Center: ({center_x.value}\",{center_y.value}\")\nBoxes 12\', {layers_str}\n{angle_str}'

        # Add the text to the plot.
        if len(layers) > 0:
            t = ax.text(text_x, text_y, text_str, color='red',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='red'))
        

def mark_poi(coord: Quantity, ax: plt.Axes, frame: sunpy.map, **kwargs):
    """
    Plots points of interest on a sunpy map.
    
    Parameters
    ----------
    coord : Quantity
        A Quantity containing a coordinate pair
    ax : matplotlib.pyplot axes object
        The axes to be plotted on.
    frame : sunpy map
        Coordinate frame being used.
    kwargs : keywords
        Keywords for specifying the mark properties.
        
    Returns
    -------
    None.
    """
    default_kwargs = {'marker': 'x', 'color': 'r'}
    kwargs = {**default_kwargs, **kwargs}

    sky = SkyCoord(*coord, frame=frame.coordinate_frame)
    p = ax.plot_coord(sky, **kwargs)


def mark_psp(coord: Quantity, ax: plt.Axes, frame: sunpy.map, **kwargs):
    """
    Plots a square to represent PSP's location.
    
    Parameters
    ----------
    x,y : list
        List of the x and y coordinates of the points of interest.
    ax : matplotlib.pyplot axes object
        The axes to be plotted on.
    frame : sunpy map
        Coordinate frame being used.
    kwargs : keywords
        Keywords for specifying the mark properties.
        
    Returns
    -------
    None.
    """
    default_kwargs = {'marker': 's', 'color': 'orange'}
    kwargs = {**default_kwargs, **kwargs}

    mark_poi(coord, ax, frame, **kwargs)

    # Determine the position of the text box.
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    x_mid, y_mid = (X_MAX+X_MIN)/2, (Y_MAX+Y_MIN)/2
    text_x = ax_xlim[1] * (x_mid-X_MIN+coord[0].to(u.arcsecond).value)/(X_MAX-X_MIN)
    text_y = ax_ylim[1] * (y_mid-Y_MIN+coord[1].to(u.arcsecond).value)/(Y_MAX-Y_MIN) + 50
    ax.text(text_x, text_y, 'PSP', color='k',
            horizontalalignment='center', verticalalignment='top', size=8)