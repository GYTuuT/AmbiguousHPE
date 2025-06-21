"""
A simple package of visiualize 2D and 3D plots in web.
author: GeYuting
"""
import re
import threading
import warnings
import webbrowser
from base64 import b64encode
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import plotly.graph_objects as go
from numpy import ndarray
from plotly.subplots import make_subplots

__all__ = ['BrowserPlot']

# --- Constants ---
WEB_STYLE = """
<html>
    <body style="background-color:282c34;">
    </body>
</html>
"""
HTML_BUFFER_SIZE = 1024 * 1024
MAX_HTML_SIZE_MB_WARNING = 30
SERVER_TIMEOUT_SECONDS = 5
LOCALHOST = "127.0.0.1"

# --- Helper Classes and Functions ---
class _CustomHTTPServer(HTTPServer):
    allow_reuse_address = True
    def __init__(self, server_address, RequestHandlerClass, html_content, stop_event):
        super().__init__(server_address, RequestHandlerClass)
        self.html_content = html_content
        self.stop_event = stop_event

class _OneShowRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(self.server.html_content)
        self.server.stop_event.set() # Signal server to stop

    def log_message(self, format, *args):
        pass # Suppress logging

def _open_html_in_web(html: str, port: Optional[int] = None, style_css: str = WEB_STYLE, autoraise: bool = True):
    """Opens HTML content in browser using a local HTTP server."""
    combined_html = (style_css + html).encode("utf8")
    html_size_mb = len(combined_html) / (1024 * 1024)
    if html_size_mb > MAX_HTML_SIZE_MB_WARNING:
        warnings.warn(f"HTML size exceeds {MAX_HTML_SIZE_MB_WARNING}MB. Consider saving to file ('save_html') for large plots.")

    stop_event = threading.Event()
    server_address = (LOCALHOST, port if port is not None else 0)
    server = _CustomHTTPServer(server_address, _OneShowRequestHandler, combined_html, stop_event) # Create server instance

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    actual_port = server.server_port # Get actual port

    webbrowser.get().open(f"http://{LOCALHOST}:{actual_port}", autoraise=autoraise)
    server_thread.join(timeout=SERVER_TIMEOUT_SECONDS) # Wait for request handling
    stop_event.set() # Ensure stop event is set
    server.shutdown() # Shutdown server immediately
    server.server_close() # Close server socket
    return actual_port


""" ------------------------------------------ """
""" Style Settings """

_fig_layout_style = {
    "autosize": True,
    "title": {},
    "margin": {"autoexpand": True, "b": 10, "t": 10, "l": 10, "r": 10},
    "font": {"family": "Times New Roman", "size": 12},
    "paper_bgcolor": 'rgba(0, 0, 0, 0)',  # transparent
    "plot_bgcolor": 'rgba(0, 0, 0, 0)',   # transparent
}

_xy_layout_axis_style = dict(
    constrain="domain",
    color='#cadee2',
    tickfont=dict(color='#cadee2'),
    zeroline=True,
)

_image_layout_axis_style = dict(
    constrain="domain",
    color='#cadee2',
    tickfont=dict(color='#cadee2'),
    showgrid=False,
    showline=False,
    zeroline=False,
    showdividers=False,
)

def _add_axis_scale_equal(axis_idx: int, flag: str = 'xy',
                          xaxis_style: Dict = None,
                          yaxis_style: Dict = None):
    """Adds axis scaling to enforce equal aspect ratio."""

    new_xaxis_style = {}
    new_yaxis_style = {}

    if xaxis_style is None and yaxis_style is None:
        if flag == 'xy':
            xaxis_style = _xy_layout_axis_style.copy()
            yaxis_style = _xy_layout_axis_style.copy()
        elif flag == 'map':
            xaxis_style = _image_layout_axis_style.copy()
            yaxis_style = _image_layout_axis_style.copy()
        else:
            raise ValueError(f"Invalid flag: {flag}. Flag must be 'xy' or 'map'.")
    else:
        new_xaxis_style = xaxis_style.copy() if xaxis_style else {}
        new_yaxis_style = yaxis_style.copy() if yaxis_style else {}

    idx_str = str(axis_idx)
    new_xaxis_style.update(scaleanchor=f'y{idx_str}', scaleratio=1)
    new_yaxis_style.update(scaleanchor=f'x{idx_str}', scaleratio=1)

    if flag == 'map':
        new_yaxis_style.update(autorange='reversed')
    return new_xaxis_style, new_yaxis_style



_scene_layout_axis_style = dict(
    backgroundcolor='#454c5a',
    color='#cadee2',
    gridcolor='#9caccc',
    tickfont=dict(color='#cadee2'),
    zeroline=False
    )

_scene_default_style = dict(
    aspectmode="manual",
    aspectratio=dict(x=1, y=1, z=1), # 3D aspect
    bgcolor='rgba(0,0,0,0)',
    # camera = {},
    xaxis=_scene_layout_axis_style,
    yaxis=_scene_layout_axis_style,
    zaxis=_scene_layout_axis_style,
)


def _adjust_scene_bounds(plot_center:ndarray=None, # [3,]
                         plot_bounds:ndarray=None, # [3,]
                         scene_layout_style:Dict=None) -> None:

    if scene_layout_style is None:
        scene_layout_style = _scene_default_style.copy()

    if (plot_center is None) or (plot_bounds is None):
        return scene_layout_style.copy()

    # set scene bound
    bound = plot_bounds.max() # to keep equal aspect.
    plot_min = plot_center - bound
    plot_max = plot_center + bound
    bounds = np.stack([plot_min, plot_max], axis=0).T

    # preserve old axis range.
    x_range, y_range, z_range = bounds
    old_xrange, old_yrange, old_zrange = (
        scene_layout_style.get('xaxis_range', None),
        scene_layout_style.get('yaxis_range', None),
        scene_layout_style.get('zaxis_range', None),
    )

    if old_xrange is not None:
        x_range[0] = min(x_range[0], old_xrange[0])
        x_range[1] = max(x_range[1], old_xrange[1])
    if old_yrange is not None:
        y_range[0] = min(y_range[0], old_yrange[0])
        y_range[1] = max(y_range[1], old_yrange[1])
    if old_zrange is not None:
        z_range[0] = min(z_range[0], old_zrange[0])
        z_range[1] = max(z_range[1], old_zrange[1])

    new_center = [(x_range[0] + x_range[1]) / 2,
                  (y_range[0] + y_range[1]) / 2,
                  (z_range[0] + z_range[1]) / 2]
    new_bounds = max(x_range[1] - x_range[0],
                     y_range[1] - y_range[0],
                     z_range[1] - z_range[0])
    x_range = [new_center[0] - new_bounds / 2, new_center[0] + new_bounds / 2]
    y_range = [new_center[1] - new_bounds / 2, new_center[1] + new_bounds / 2]
    z_range = [new_center[2] - new_bounds / 2, new_center[2] + new_bounds / 2]

    new_scene_layout_style = scene_layout_style.copy()
    new_scene_layout_style.update(
        dict(
            xaxis_autorange=False,
            yaxis_autorange=False,
            zaxis_autorange=False,
            xaxis_range=x_range,
            yaxis_range=y_range,
            zaxis_range=z_range,
            )
        )

    return new_scene_layout_style


""" ------------------------------------------ """



## ============
class OneSubplot(NamedTuple):
    idx : int = 1
    pos : List = [1, 1]
    layout_type : str = 'xy'

    plot_type : str = 'map' # xy, map, xyz: 'xy' represent 2D,  'map' represent image that should reverse Y and 'xyz' represet 3D.
    plot_center : ndarray=None # the center of plot axis, [3,] for 3D, and [2, ] for 2D
    plot_bounds : ndarray=None # the bounds of plot axis, [3,] for 3D, and [2, ] for 2D

    trace : go.Trace=None


## =============
class BrowserPlot: # Topdown Structure: Fig -> Subplot('xy' or 'scene') -> Trace

    _subplots : List[OneSubplot]
    _grid : List

    def __init__(self, nrows:int=1, ncols:int=1) -> None:

        assert (nrows >= 1) and (ncols >= 1)
        nrows, ncols = int(nrows), int(ncols)

        self.num_subplots = nrows * ncols
        self._grid = [nrows, ncols]
        self._subplots = []


    # ---------
    def check_subplot_pos(self, grid_pos:List):

        row, col = grid_pos[0], grid_pos[1]
        if (row < 1) or (row > self._grid[0]) or (col < 1) or (col > self._grid[1]):
            raise ValueError(f'Input pos {grid_pos} does not exist when grid shape is {self._grid}, and starting pos of grid is [1, 1]')

        return (row - 1) * self._grid[1] + col # subplot_idx


    # ---------
    def image(self, image:ndarray, pos:List=[1, 1], name:str=''):
        """
        Add image to subplot

        Parameters
        ---
        pos:
            The plots' grid position `[row, col]` to embed image.
        name: str
            inserted image's name
        """

        layout_type = 'xy'
        plot_type = 'map'
        subplot_idx = self.check_subplot_pos(pos)

        img_shape = image.shape
        if len(img_shape) == 2: # depth map
            raise ValueError(f'Current method does not support depth image, try method .heatmap')

        elif len(img_shape) == 3: # rgb, rgba

            format_check = ((image.dtype != np.uint8) & (image.min() >= 0.) & (image.max() <= 1.)) \
                         | ((image.dtype == np.uint8) & (image.min() >= 0) & (image.max() <= 255))
            if not format_check:
                raise ValueError('Invalid image format, need float [0.0, 1.0] or uint8 [0, 255].')

            if img_shape[-1] == 3: # rgb
                color_mode = 'rgb'
            elif img_shape[-1] == 4: # rgba
                color_mode = 'rgba'
            else:
                raise ValueError('Unsupport input shape, only RGB or RGBA.')

        else:
            raise ValueError(f'Unknown image shape {img_shape}.')

        img_uint8 = (image * 255).astype(np.uint8) if (image.dtype != np.uint8) else image

        # Use JPEG compression with quality=80
        import io

        from PIL import Image
        img = Image.fromarray(img_uint8)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        data = buf.getvalue()

        trace = go.Image(name=name, source="data:image/jpeg;base64," + b64encode(data).decode('utf-8'))

        plot = OneSubplot(idx=subplot_idx, pos=pos, layout_type=layout_type, plot_type=plot_type, trace=trace)
        self._subplots.append(plot)


    # ------------
    def heatmap(self, heatmap:ndarray, pos:List=[1, 1], cmap:str='greys', name:str=''):
        """
        Add heatmap to subplot

        Parameters
        ---
        heatmap: ndarray
            `[H, W]`, float or int data.
        pos: List
            The plots' grid position `[row, col]` to embed image.
        cmap: str
            color map type
        name: str
            inserted heatmap's name

        """

        layout_type = 'xy'
        plot_type = 'map'
        subplot_idx = self.check_subplot_pos(pos)

        if len(heatmap.shape) != 2: # depth map
            raise ValueError(f'Current method does only support depth image with 2 dimensions.')

        _Cmaps = ['greys', 'purples', 'blues', 'greens', 'oranges', 'reds',
                  'cividis', 'viridis', 'magma', 'earth']

        if not (cmap in _Cmaps):
            raise ValueError(f'Unsupport color map, only cmap in {_Cmaps}')

        heatmap = heatmap.astype(np.float32)
        trace = go.Heatmap(name=name, z=heatmap,
                           autocolorscale=False, colorscale=cmap,
                           showscale=False, showlegend=False)

        plot = OneSubplot(idx=subplot_idx, pos=pos, layout_type=layout_type, plot_type=plot_type, trace=trace)
        self._subplots.append(plot)


    # ------------
    def scatter2D(self, xys:ndarray, pos:List=[1, 1], name:str='',
                  size:float=5., color:str='#98F5FF', max_points:int=None):
        """
        Scatter the input 2D points in subplot.

        Parameters
        ----------
        xys: ndarray
            `[N, 2]`, XY coordinates of 2D points.
        pos: List
            The plots' grid position `[row, col]` to embed image.
        name: str
            inserted scatter's name.
        size: float
            pixel size of one point marker.
        color: str
            color in HEX format.
        max_points: int, optional
            Maximum number of points to plot. If the number of points exceeds this value, a random subset of points will be plotted.
        """
        layout_type = 'xy'
        plot_type = 'xy'
        subplot_idx = self.check_subplot_pos(pos)

        if not ((xys.ndim == 2) and (xys.shape[-1] == 2)):
            raise ValueError(f'Invalid input xys array shape, only support [N, 2]')

        if max_points is not None and xys.shape[0] > max_points:
            indices = np.random.choice(xys.shape[0], max_points, replace=False)
            xys = xys[indices].astype(np.float32)
        else:
            xys = xys.astype(np.float32)

        trace = go.Scatter(name=name, x=xys[:, 0], y=xys[:, 1], mode='markers',
                           marker=dict(color=color, size=size, symbol='circle'),
                           showlegend=False)

        plot = OneSubplot(idx=subplot_idx, pos=pos, layout_type=layout_type, plot_type=plot_type, trace=trace)
        self._subplots.append(plot)


    # ------------
    def plot2D(self, xys:ndarray, pos:List=[1, 1], name:str='',
               thickness:float=5., color:str='#66CD00',
               markout:bool=False, marker_size:float=10., marker_color:str='#458B00',):
        """
        Plot the lines between input 2D points.

        Parameters
        ----------
        xys: ndarray
            `[N, 2]`, XY coordinates of 2D points.
        pos: List
            The plots' grid position `[row, col]` to embed image.
        name: str
            inserted scatter's name.
        thickness: float
            pixel width of plot lines.
        color: str
            line color in HEX.
        markout: bool
            whether to mark the 2D point out.
        marker_size: float
            pixel size of point marker, only effective when `markout=True`
        marker_color: str
            marker color in HEX, only effective when `markout=True`
        """
        layout_type = 'xy'
        plot_type = 'xy'
        subplot_idx = self.check_subplot_pos(pos)

        if not ((xys.ndim == 2) and (xys.shape[-1] == 2)):
            raise ValueError(f'Invalid input xys array shape, only support [N, 2]')

        xys = xys.astype(np.float32)

        trace = go.Scatter(name=name, x=xys[:, 0], y=xys[:, 1], mode='lines+markers' if markout else 'lines',
                           line=dict(color=color, width=thickness, dash='solid', simplify=False),
                           marker=dict(color=marker_color, size=marker_size, symbol='circle') if markout else None,
                           showlegend=False)

        plot = OneSubplot(idx=subplot_idx, pos=pos, layout_type=layout_type, plot_type=plot_type, trace=trace)
        self._subplots.append(plot)


    # ------------
    def scatter3D(self, xyzs:ndarray, pos:List=[1, 1], name:str='',
                  size:float=5., color:str='#98F5FF', max_points:int=None):
        """
        Scatter the input 3D points in subplot.

        Parameters
        ----------
        xyzs: ndarray
            `[N, 3]`, XYZ coordinates of 3D points.
        pos: List
            The plots' grid position `[row, col]` to embed image.
        name: str
            inserted scatter's name.
        size: float
            pixel size of one point marker.
        color: str
            color in HEX format.
        max_points: int, optional
            Maximum number of points to plot. If the number of points exceeds this value, a random subset of points will be plotted.
        """
        layout_type = 'scene'
        plot_type = 'xyz'
        subplot_idx = self.check_subplot_pos(pos)

        if not ((xyzs.ndim == 2) and (xyzs.shape[-1] == 3)):
            raise ValueError(f'Invalid input xyzs array shape, only support [N, 3]')

        if max_points is not None and xyzs.shape[0] > max_points:
            indices = np.random.choice(xyzs.shape[0], max_points, replace=False)
            xyzs = xyzs[indices].astype(np.float32)
        else:
            xyzs = xyzs.astype(np.float32)

        trace = go.Scatter3d(name=name, x=xyzs[:, 0], y=xyzs[:, 1], z=xyzs[:, 2],
                             mode='markers', marker=dict(color=color, size=size, symbol='circle'),
                             showlegend=False)

        plot = OneSubplot(idx=subplot_idx, pos=pos, layout_type=layout_type,
                          plot_type=plot_type,
                          plot_center=xyzs.mean(0),
                          plot_bounds=xyzs.max(0) - xyzs.min(0),
                          trace=trace)
        self._subplots.append(plot)


    # ------------
    def plot3D(self, xyzs:ndarray, pos:List=[1, 1], name:str='',
               thickness:float=5., color:str='#66CD00',
               markout:bool=False, marker_size:float=10., marker_color:str='#458B00',):
        """
        Plot the lines between input 3D points.

        Parameters
        ----------
        xyzs: ndarray
            `[N, 3]`, XYZ coordinates of 3D points.
        pos: List
            The plots' grid position `[row, col]` to embed image.
        name: str
            inserted scatter's name.
        thickness: float
            pixel width of plot lines.
        color: str
            line color in HEX.
        markout: bool
            whether to mark the 3D point out.
        marker_size: float
            pixel size of point marker, only effective when `markout=True`
        marker_color: str
            marker color in HEX, only effective when `markout=True`
        """
        layout_type = 'scene'
        plot_type = 'xyz'
        subplot_idx = self.check_subplot_pos(pos)

        if not ((xyzs.ndim == 2) and (xyzs.shape[-1] == 3)):
            raise ValueError(f'Invalid input xyzs array shape, only support [N, 3]')

        xyzs = xyzs.astype(np.float32)

        trace = go.Scatter3d(name=name, x=xyzs[:, 0], y=xyzs[:, 1], z=xyzs[:, 2],
                                 mode='lines+markers' if markout else 'lines',
                                 line=dict(color=color, width=thickness, dash='solid'),
                                 marker=dict(color=marker_color, size=marker_size, symbol='circle') if markout else None,
                                 showlegend=False)

        plot = OneSubplot(idx=subplot_idx, pos=pos, layout_type=layout_type,
                          plot_type=plot_type,
                          plot_center=xyzs.mean(0),
                          plot_bounds=xyzs.max(0) - xyzs.min(0),
                          trace=trace)
        self._subplots.append(plot)


    # --------------
    def mesh3D(self, verts:ndarray, faces:ndarray, verts_color:ndarray=None,
               pos:List=[1, 1], name:str=''):
        """
        Show the 3D mesh textured with its texture.

        Parameters:
        ----
        verts: ndarray
            `[N, 3]`, vertices of mesh.
        faces: ndarray
            `[F, 3]`, faces of mesh.
        verts_color: ndarray
            `[N, 3]`, color of vertices, if None, use the verts normals.
        pos: List
            The plots' grid position `[row, col]` to embed image.
        name: str
            inserted scatter's name.
        """

        layout_type = 'scene'
        plot_type = 'xyz'
        subplot_idx = self.check_subplot_pos(pos)

        if not ((verts.ndim == 2) and (verts.shape[-1] == 3)):
            raise ValueError(f'Invalid input verts array shape, only support [N, 3]')

        if not ((faces.ndim == 2) and (faces.shape[-1] == 3)):
            raise ValueError(f'Invalid input faces array shape, only support [N, 3]')

        # replace the unused verts
        verts_used = np.zeros([len(verts), ]).astype(np.bool_)
        verts_used[np.unique(faces)] = True
        verts_center = verts[verts_used].mean(0)
        verts[~verts_used] = verts_center
        plot_bounds = verts[verts_used].max(0) - verts[verts_used].min(0)

        if verts_color is None:
            verts_color = self.compute_vertex_normals(verts, faces)

        trace = go.Mesh3d(name=name,
                          x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                          vertexcolor=verts_color,
                          showlegend=False, showscale=False)


        plot = OneSubplot(idx=subplot_idx, pos=pos, layout_type=layout_type,
                          plot_type=plot_type,
                          plot_center=verts[verts_used].mean(0),
                          plot_bounds=plot_bounds,
                          trace=trace)
        self._subplots.append(plot)


    # --------------
    @staticmethod
    def compute_vertex_normals(verts:np.ndarray, faces:np.ndarray) -> np.ndarray:
        """
        verts: ndarray
            `[N, 3]`
        faces: ndray
            `[F, 3]`
        verts_normal: ndarray
            `[N, 3]`
        """
        verts_normal = np.zeros_like(verts)

        verts_per_face = verts[faces.astype(np.int32)] # (F, 3, 3)
        faces_normal = np.cross(
            verts_per_face[:, 2] - verts_per_face[:, 1],
            verts_per_face[:, 0] - verts_per_face[:, 1],
            axis=-1) # [F, 3]

        for i in range(3):
            verts_normal[faces[:, i]] += faces_normal

        verts_normal = verts_normal / np.clip(np.linalg.norm(verts_normal, axis=-1, keepdims=True), a_min=1e-6, a_max=None)

        return verts_normal



    # ------------
    def embed_trace_to_fig(self) -> go.Figure:

        # 1. create subplots and named grid layout
        subplots_specs = [[None for _ in range(self._grid[1])]  for _ in range(self._grid[0])]
        for sp in self._subplots:
            spec = subplots_specs[sp.pos[0]-1][sp.pos[1]-1] # old spec
            if spec is not None:
                if spec['type'] != sp.layout_type: # conflict subplot type
                    raise TypeError(f'Subplot at grid {sp.pos} can not show 2D and 3D data at the same time.')
                else:
                    continue
            else:
                subplots_specs[sp.pos[0]-1][sp.pos[1]-1] = {'type':sp.layout_type}

        grid_layout_names = [[None for _ in range(self._grid[1])]  for _ in range(self._grid[0])]
        xy_idx, scene_idx = 0, 0
        for r in range(self._grid[0]):
            for c in range(self._grid[1]):
                spec = subplots_specs[r][c]
                if spec is None: # not-used subplot
                    continue
                if spec['type'] == 'xy':
                    xy_idx += 1
                    grid_layout_names[r][c] = f'axis{xy_idx}'
                elif spec['type'] == 'scene':
                    scene_idx += 1
                    grid_layout_names[r][c] = f'scene{scene_idx}'
                else:
                    ...
        fig = make_subplots(*self._grid, specs=subplots_specs)

        # 2. set layout style
        layout_style = _fig_layout_style.copy()
        grid_has_style = np.zeros(self._grid).astype(np.bool_)
        for sp in self._subplots:
            has_set = grid_has_style[sp.pos[0]-1, sp.pos[1]-1]
            grid_type = subplots_specs[sp.pos[0]-1][sp.pos[1]-1]['type']
            name = grid_layout_names[sp.pos[0]-1][sp.pos[1]-1]

            if grid_type == 'xy':
                if has_set: # preserve previous settings
                    xaxis_style = layout_style[f'x{name}'].copy()
                    yaxis_style = layout_style[f'y{name}'].copy()
                    xaxis_style, yaxis_style = _add_axis_scale_equal(int(re.findall(r'\d+', name)[0]), sp.plot_type,
                                                                     xaxis_style=xaxis_style,
                                                                     yaxis_style=yaxis_style)
                else:
                    xaxis_style, yaxis_style = _add_axis_scale_equal(int(re.findall(r'\d+', name)[0]), sp.plot_type)
                    grid_has_style[sp.pos[0]-1, sp.pos[1]-1] = True

                layout_style.update({f'x{name}':xaxis_style,
                                     f'y{name}':yaxis_style})

            if grid_type == 'scene':
                if has_set:
                    scene_style = layout_style[name].copy()
                    scene_style = _adjust_scene_bounds(sp.plot_center, sp.plot_bounds,
                                                       scene_layout_style=scene_style)
                else:
                    scene_style = _adjust_scene_bounds(sp.plot_center, sp.plot_bounds)
                    grid_has_style[sp.pos[0]-1, sp.pos[1]-1] = True

                layout_style.update({name:scene_style})

        fig.update_layout(**layout_style)

        # 3. add traces
        for sp in self._subplots:
            if sp.trace is None:
                continue
            fig.add_trace(sp.trace, row=sp.pos[0], col=sp.pos[1])

        return fig


    # --------
    def show(self, port:int=None):
        fig = self.embed_trace_to_fig()
        _open_html_in_web(fig.to_html(), port=port)


    # --------
    def save_html(self, path:str):
        if path[-5:] != '.html':
            path = path + '.html'
        fig = self.embed_trace_to_fig()
        fig.write_html(path)
