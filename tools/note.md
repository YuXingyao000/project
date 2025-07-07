# The pipeline to sample the points on the surface of the solid
## Training
1. Sample 8192 points on the solid as GT.
2. Randomly choose n from 2048 to 6144.
3. Randomly choose a viewpoint from 16 viewpoints.
4. Remove n furthest points from the viewpoint.
5. Downsample the remaining point clouds to 2048 points as the input data for models. 

## Evaluation
1. Choose a viewpoint from 8 viewpoints.
2. Randomly choose n as 2048, 4096 or 6144.
3. Randomly choose a viewpoint from 16 viewpoints.
4. Remove n furthest points from the viewpoint.
5. Downsample the remaining point clouds to 2048 points as the input data for models. 
 