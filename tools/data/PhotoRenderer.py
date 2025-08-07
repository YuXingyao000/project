import open3d as o3d

import os

import numpy as np
from OCC.Core.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NameOfMaterial_Silver, Graphic3d_TypeOfReflection
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Display.OCCViewer import Viewer3d

from PIL import Image
from scipy.spatial.transform import Rotation


class ShapeRenderer(Viewer3d):
    """The offscreen renderer is inherited from Viewer3d.
    The DisplayShape method is overridden to export to image
    each time it is called.
    """

    def __init__(self, screen_size=(224, 224), color=[255, 255, 255]):
        super().__init__()
        # create the renderer
        self.Create(display_glinfo=False)
        self.SetSize(screen_size[0], screen_size[1])
        self.SetModeShaded()
        self.set_bg_gradient_color(color, color)
        self.capture_number = 0

class PhotoRenderer:
    """Renders 3D shapes from multiple viewpoints to generate image datasets."""
    
    def __init__(self, shape, screen_size=(224, 224)):
        """
        Initialize the photo renderer.
        
        Args:
            - shape: OCCT shape to render
            - screen_size: Size of the rendered image
        """
        self.shape = shape
        self.screen_size = screen_size
        self.renderer = self.init_renderer(screen_size)
        
    def process(self):
        """Render the shape from 64 systematic viewpoints.
        
        Returns:
            - List of numpy arrays containing rendered images (224x224x3)
        """
        # Generate systematic viewpoints
        viewpoints, up_directions = self.generate_viewpoints()
        
        # Render images from all viewpoints
        images = self.render_images(viewpoints, up_directions, Graphic3d_NameOfMaterial_Silver)
        
        return images, viewpoints, up_directions
    
    def init_renderer(self, screen_size=(224, 224), color=[255, 255, 255]):
        """Initialize the offscreen renderer."""
        renderer = ShapeRenderer(screen_size=screen_size, color=color)
        
        renderer.camera.SetProjectionType(1)
        renderer.camera.SetFOVy(45)
        renderer.camera.SetAspect(1)
        
        return renderer
    
    def setup_material(self, enum_material=Graphic3d_NameOfMaterial_Silver):
        """Setup material properties for the shape."""
        material = Graphic3d_MaterialAspect(enum_material)
        material.SetReflectionModeOff(Graphic3d_TypeOfReflection.Graphic3d_TOR_SPECULAR)
        return material
    
    def render_images(self, viewpoints, up_directions, material):
        """Generate images from given viewpoints around the object.
        
        Args:
            viewpoints: List of (view_position, up_direction) tuples or just view positions
            
        Returns:
            List of numpy arrays containing rendered images
        """
        images = []
        material = self.setup_material(material)
        self.renderer.DisplayShape(self.shape, material=material, update=True)
        
        for i, viewpoint in enumerate(viewpoints):
            self.renderer.camera.SetEyeAndCenter(gp_Pnt(viewpoint[0], viewpoint[1], viewpoint[2]), gp_Pnt(0., 0., 0.))
            self.renderer.camera.SetUp(gp_Dir(up_directions[i][0], up_directions[i][1], up_directions[i][2]))
            data = self.renderer.GetImageData(224, 224)
            img = Image.frombytes("RGB", (224, 224), data)
            img = np.asarray(img)
            img = img[::-1]
            # Image.fromarray(img).save(f"debug_{i}.png")
            images.append(img)
        return images
    
    
    def generate_viewpoints(self):
        """Generate augmented viewpoints
        It is sampling viewpoints from a cube.
        Rotation (90 degrees) and mirror(horizontal) are applied to the viewpoints.
        
        Args:
            num_viewpoints: Number of viewpoints to generate (default: 64)
            
        Returns:
            List of view_position
        """
        viewpoints = []
        up_directions = []
        init_pos = np.array((2, 2, 2))
        up_pos = np.array((2, 2, 3))
        for i in range(64):
            angles = np.array([
                i % 4,
                i // 4 % 4,
                i // 16
            ])
            
            # Rotation (90 degrees) and mirror(horizontal) is realized by applying rotation matrix to the up direction
            matrix = Rotation.from_euler('xyz', angles * np.pi / 2).as_matrix().T
            view = (matrix @ init_pos.T).T
            up = (matrix @ up_pos.T).T
            viewpoints.append(view)
            up_directions.append(up)

        return viewpoints, up_directions
    


if __name__ == "__main__":
    from tools.data.ABCReader import ABCReader
    from tools.data.SolidProcessor import SolidProcessor
    
    for folder_name in os.listdir("D:/XiaoLunWen/data/abc_data"):
        reader = ABCReader()
        file_name = os.listdir(f"D:/XiaoLunWen/data/abc_data/{folder_name}")[0]
        reader.read_step_file(f"D:/XiaoLunWen/data/abc_data/{folder_name}/{file_name}")
        if reader.num_solids == 0:
            continue
        elif reader.num_solids > 1:
            print(f"Multiple solids found in {folder_name}")
            continue
        shape = reader.get_solids()[0]
        solid_processor = SolidProcessor(shape)
        solid_processor.normalize_shape()
        # debug_vis(solid_processor.solid)
        renderer = PhotoRenderer(solid_processor.solid)
        svr_imgs = renderer.process()
