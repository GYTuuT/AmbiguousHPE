

"""NOTE NEED TO INSTALL FFMPEG PACKAGE FIRSTLY"""

import os

## -----------------------------------
def do_IMG2GIF(absPath:str, gif_name:str='out', fps=20, resolution:list=[512,512], suffix:str='.png'):
    """
    Compose images into GIF with high quality using ffmpeg package.

    Parameters
    ---
    absPath : str 
        absolute path of the image directory.
    gif_name : str 
        name of output gif file.
    fps : int
        frames per second.
    resolution : List
        `[H, W]`.
    suffix : str
        suffix of filename, such as '.png', '.jpg' ...
        
    """
    W, H = resolution[1], resolution[0]
    cmd = f'ffmpeg -y -f image2 -r {fps} -s {W}x{H} -pattern_type glob -i \'{absPath}/*{suffix}\' -b:v 16384k {absPath}/temp.mp4'\
        + '&&' + f'ffmpeg -y -i {absPath}/temp.mp4 -b:v 16384k -vf fps={fps},scale={W}x{H}:flags=lanczos,palettegen {absPath}/palette.png'\
        + '&&' + f'ffmpeg -y -i {absPath}/temp.mp4 -i {absPath}/palette.png -b 16384k -s {W}x{H} -filter_complex \"fps={fps}, scale={W}x{H}:flags=lanczos[x];[x][1:v]paletteuse\" {absPath}/{gif_name}.gif'\
        + '&&' + f'rm {absPath}/palette.png {absPath}/temp.mp4'
    os.popen(cmd)