import open3d as o3d

import os
import random
import shutil
from pathlib import Path

import ray, trimesh
import numpy as np
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NameOfMaterial_Silver, Graphic3d_TypeOfShadingModel, \
    Graphic3d_NameOfMaterial_Brass, Graphic3d_TypeOfReflection, Graphic3d_AspectLine3d
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt, gp_Dir
from OCC.Display.OCCViewer import OffscreenRenderer, Viewer3d
from OCC.Extend.DataExchange import read_step_file, write_step_file
import traceback, sys

from PIL import Image
from scipy.spatial.transform import Rotation

from tools.data.ViewpointSampler import ViewpointSampler

class MyOffscreenRenderer(Viewer3d):
    """The offscreen renderer is inherited from Viewer3d.
    The DisplayShape method is overridden to export to image
    each time it is called.
    """

    def __init__(self, screen_size=(224, 224)):
        super().__init__()
        # create the renderer
        self.Create(display_glinfo=False)
        self.SetSize(screen_size[0], screen_size[1])
        self.SetModeShaded()
        self.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
        self.capture_number = 0

class PhotoRenderer:
    def __init__(self, shape, strategy='cube', n_viewpoints=None, radius=1.0, n_rays=2048):
        self.shape = shape
        self.strategy = strategy
        self.n_viewpoints = n_viewpoints
        self.radius = radius
        self.n_rays = n_rays
        self.viewpoints = ViewpointSampler(strategy, n_viewpoints, radius).sample_viewpoints()
    
    def render(self):
        # Single view
        display = MyOffscreenRenderer()
        mat = Graphic3d_MaterialAspect(Graphic3d_NameOfMaterial_Silver)
        mat.SetReflectionModeOff(Graphic3d_TypeOfReflection.Graphic3d_TOR_SPECULAR)
        display.DisplayShape(self.shape, material=mat, update=True)
        display.camera.SetProjectionType(1)
        # display.View.Dump(str(img_root / v_folder / f"view_.png"))
        svr_imgs = []
        display.camera.SetEyeAndCenter(gp_Pnt(2., 2., 2.), gp_Pnt(0., 0., 0.))
        display.camera.SetUp(gp_Dir(0,0,1))
        display.camera.SetAspect(1)
        display.camera.SetFOVy(45)
        init_pos = np.array((2, 2, 2))
        up_pos = np.array((2, 2, 3))
        for i in range(64):
            angles = np.array([
                i % 4,
                i // 4 % 4,
                i // 16
            ])
            matrix = Rotation.from_euler('xyz', angles * np.pi / 2).as_matrix().T
            view = (matrix @ init_pos.T).T
            up = (matrix @ up_pos.T).T
            display.camera.SetEyeAndCenter(gp_Pnt(view[0], view[1], view[2]), gp_Pnt(0., 0., 0.))
            display.camera.SetUp(gp_Dir(up[0], up[1], up[2]))
            data = display.GetImageData(224, 224)
            img = Image.frombytes("RGB", (224, 224), data)
            img1 = np.asarray(img)
            img1 = img1[::-1]
            svr_imgs.append(img1)
        return svr_imgs
        