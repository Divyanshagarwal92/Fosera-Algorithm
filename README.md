#Focal Sweep Camera (Fosera) for Space Time Refocussing

Focal stack of images over small duration of time were captured to estimate
a depth map. Depth map was used as an index map to facilitate post-focusing
over space and timei to create "breathing pictures". [C++, OpenCV]

A UI tool is included to dynamically refocus an image using space-time refocussing
by clicking different locations in the image.


#Steps to recreate results:
1. Calculate Reliable Depth Map
  ./depthMap <in_dir> <out_dir> <debug_mode> <num_images>

2. Image segmentation
  ./segment 0.1 600 20 <out_dir>/averageImage.ppm <out_dir>/averageImage-seg.ppm

3. Hole filling
  ./holeFilling <out_dir>/averageImage-seg.ppm <out_dir>/reliableMap.ppm $2 $4


4. UI
   ./ui $1 $2 $4

#Relevant Research:
http://www1.cs.columbia.edu/CAVE/publications/pdfs/Zhou_TR12

