echo "Calculate Reliable Depth Map "
./depthMap_v2 $1 $2 $3 $4

echo ""
echo "Image segmentation"
./segment 0.1 600 20 $2/averageImage.ppm $2/averageImage-seg.ppm

echo ""
echo "Hole filling"
./holeFilling $2/averageImage-seg.ppm $2/reliableMap.ppm $2 $4


echo ""
echo "UI"
./ui $1 $2 $4
