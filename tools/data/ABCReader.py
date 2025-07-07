import os
from pathlib import Path

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND
from OCC.Core.TopoDS import topods
from OCC.Core.TopExp import TopExp_Explorer

from tools.data.SolidWrapper import SolidWrapper

def debug_vis(shape):
    try:
        from OCC.Display.SimpleGui import init_display
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(shape, update=True)
        print("Shape visualization displayed. Close the window to continue.")
        start_display()
    except Exception as viz_error:
        print(f"Visualization error: {str(viz_error)}")

class ABCReader:
    """
    Read raw data from ABC dataset.
    A class to read and process STEP files using python-OCC
    Supports multiple solids in a single STEP file
    """
    
    def __init__(self, debug=False, visualize=False):
        self.reader = STEPControl_Reader()
        self.shape = None  
        self.solids = []   
        self.num_solids = 0
        self.is_debug = debug
        self.is_visualize = visualize
    
    def read_step_file(self, file_path):
        """
        Read a STEP file and load the shape
        
        Args:
            file_path (str): Path to the STEP file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return False
        if isinstance(file_path, Path):
            file_path = file_path.as_posix()
        
        try:
            # Read the STEP file
            status = self.reader.ReadFile(file_path)
            
            if status == IFSelect_RetDone:
                # Transfer the shape
                self.reader.TransferRoots()
                self.shape = self.reader.OneShape()
                
                # Optional visualization for debugging
                if self.is_visualize:
                    debug_vis(self.shape)
                        
                if self.shape is None:
                    if self.is_debug:
                        print("Error: No shape found in the STEP file")
                    return False
                
                print(f"Successfully loaded STEP file: {file_path}")
                
                # Extract solids and geometric elements
                self._extract_solids()
                
                return True
            else:
                print(f"Error reading STEP file: {file_path}")
                return False
                
        except Exception as e:
            print(f"Error processing STEP file: {str(e)}")
            return False
    
    def _extract_solids(self):
        """Extract individual solids from the shape"""
        assert self.shape is not None, "Shape is None"
            
        # Clear previous solids
        self.solids = []
        
        # Check if the shape is a compound (multiple solids)
        if self.shape.ShapeType() == TopAbs_COMPOUND:
            # Extract solids from compound
            solid_explorer = TopExp_Explorer(self.shape, TopAbs_SOLID)
            while solid_explorer.More():
                solid = topods.Solid(solid_explorer.Current())
                self.solids.append(solid)
                solid_explorer.Next()
        elif self.shape.ShapeType() == TopAbs_SOLID:
            # Single solid
            solid = topods.Solid(self.shape)
            self.solids.append(solid)
            
        
        self.num_solids = len(self.solids)
        if self.is_debug:
            print(f"Extracted {self.num_solids} solid(s) from STEP file")
    
    def get_solids(self):
        return self.solids


if __name__ == "__main__":
    # Debug Only
    data_root_path = Path("data/")
    
    step_files = []
    for data_dir in data_root_path.iterdir():
        if data_dir.is_dir():
            for file in data_dir.iterdir():
                step_files.append(file)
    
    total_solid = 0
    for one_step_file_path in step_files:
        reader = ABCReader()
        reader.read_step_file(one_step_file_path)
        solids = reader.get_solids()
        for solid in solids:
            s = solid.GetSolid()
            debug_vis(s)
