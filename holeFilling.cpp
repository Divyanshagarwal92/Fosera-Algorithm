#include<iostream>
#include<string>
#include <cv.h>
#include <highgui.h>

#include<algorithm>
#include<vector>
#include<ctime>
#include<math.h>
#include<bitset>
#include<chrono>
#include<utility>

using namespace std;
using namespace cv;
using namespace std::chrono;

class Pixel
{
    public:
    int pixX;
    int pixY;
    int pixZ;
    
    Pixel( int x, int y, int z)
    {
        pixX = x;
        pixY = y;
        pixZ = z;
    }
};

int NUM_FRAMES;
//Sparsity greater than 20% in a neighbourhood is considered to have unreliable information
#define SPARSITY_TH 0.4
//Specifies the kernel size for interpolation method 1, i.e borrow info from neighbours
#define KERNEL_SIZE 10

//More than 50% of the segment should have reliable information.
#define RELIABILITY_TH 0.6
int borrowFromNeighbours( Mat depthMap, int y, int x, int kernelSize)
{
    vector<int> neighbourhood;
    int cnt_sparse = 0;
    for (int i = y - kernelSize/2; i < y + kernelSize/2; i++)
    {
        for( int j = x - kernelSize/2; j < x + kernelSize/2; j++)
        {
            int value = depthMap.at<uchar>(i,j);
            //value = round(value/10);
            
            //Value scaled from 0 to NUM_FRAMES
            if(value==0)
            {
                cnt_sparse++;
                continue;
            }
            else
                neighbourhood.push_back(value);
        }
    }
    if( double(cnt_sparse)/(kernelSize*kernelSize) > SPARSITY_TH )
        return -1;
    
    int minVal = *min_element(neighbourhood.begin(), neighbourhood.end());
    int maxVal = *max_element(neighbourhood.begin(), neighbourhood.end());
    
    if (maxVal-minVal <= 3)
    {
        return round((maxVal + minVal)/2.0);
    }
    
    return -1;
}


//Interpolation method 1 - borrow info from neighbours. Only consider very strong information
Mat interpolation1( Mat relD)
{
    Mat depthMap;
	relD.copyTo(depthMap);
    
    int kernelSize = KERNEL_SIZE;
    
    for(int x = kernelSize/2; x < depthMap.rows-kernelSize/2; x++)
	{
		for( int y = kernelSize/2; y < depthMap.cols - kernelSize/2; y++)
		{
            //Scaling value to 1 to number of frames. O means no Information
            int value = depthMap.at<uchar>(x, y);
            //int value = round(depthMap.at<uchar>(x, y)/10);
            if(  value == 0)
            {
                value = borrowFromNeighbours( relD, x, y, kernelSize);
                if(value != -1)
                    depthMap.at<uchar>(x,y) = value;
            }
                
		}

	}
    return depthMap;

}

//Get Segments information - Mapping each segment to constituent pixels
// Segment 'i' - > set of all pixels (x,y) lying in that segment and having reliable depth information
//map < string, vector< Pixel > > countSegments(Mat segI, Mat depthI)
void countSegments(Mat segI, Mat depthI, map < string, vector< Pixel > > &segmentInfo, map < string, vector< Pixel > > &segmentHole )
{
    
    vector<int> myvec;
    
    cout << "Rows: "  << segI.rows << " Cols: " << segI.cols << endl;
    
    for( int x = 0; x < segI.rows; x++)
    {
        for( int y = 0; y < segI.cols; y++)
        {
            int index = depthI.at<uchar>(x,y);
            //index = round(index/10);
            index = index;
            
            //Create hash value from RGB information
            string hash = "";
            for( int ch = 0; ch < 3; ch++)
            {
                int val = segI.at<Vec3b>( x, y)[ch];
                bitset<8> bitinfo(val);
                hash = hash + bitinfo.to_string();
            }
            
            if(index == 0)
            {
                //Unreliable index information
                
                //Check whether mapping exists in map SegmentHole
                if(segmentHole.count(hash) == 0)
                {
                    Pixel pix( x, y, index);
                    vector< Pixel > newvec;
                    segmentHole[hash] = newvec;
                    segmentHole[hash].push_back(pix);
                }
                else
                {
                    Pixel pix( x, y, index);
                    segmentHole[hash].push_back(pix);
                }

             
            }
            else
            {
                //Reliable index information
                
                //Check whether has information exists in map SegmentInfo
                if(segmentInfo.count(hash) == 0)
                {
                    Pixel pix( x, y, index);
                    vector< Pixel > newvec;
                    segmentInfo[hash] = newvec;
                    segmentInfo[hash].push_back(pix);
                }
                else
                {
                    Pixel pix( x, y, index);
                    segmentInfo[hash].push_back(pix);
                }
            }
        }
    }

    //Testing segmentInfo
    cout << "Number of segmentInfo : " << segmentInfo.size() << endl;
    cout << "Number of segmentHole : " << segmentHole.size() << endl;

}


Mat planeEstimation( map < string, vector<Pixel> > &segmentInfo, map < string, vector<Pixel> > &segmentHole, Mat relD)
{
    
    Mat depthMap;
    relD.copyTo(depthMap);
    map< string, vector<Pixel> >::iterator it = segmentInfo.begin();
    
    for( ; it != segmentInfo.end(); it++)
    {

        string key = it->first;
        vector<Pixel> superPixInfo = it->second;
        vector<Pixel> superPixHole = segmentHole[key];
        
        int countReliable = superPixInfo.size();
        int countSparse = superPixHole.size();
        //cout << "# Sparse: " << countSparse << " # Reliable: " << countReliable << endl;

        double fillPercentage = double(countReliable)/(countReliable + countSparse);

        //Heuristic 1: If no sparse information, continue
        if( countSparse == 0)
            continue;
        
        //Heuristic 2: If fillpercentage < FILL_TH, continue,
        //Too much unreliable information to make a decision in the segment
        
        if( fillPercentage < RELIABILITY_TH)
           continue;
        
        //Based on superPixInfo - find the equation of plane
        Mat X = Mat::zeros(countReliable, 4, CV_32FC1);
        Mat coeff = Mat::zeros(4, 1, CV_32FC1);
        
        vector<Pixel>::iterator it2 = superPixInfo.begin();
        int i = 0;
        for(; it2 != superPixInfo.end(); it2++)
        {
            depthMap.at<uchar>(it2->pixX, it2->pixY) = (it2->pixZ);
            if (it2->pixZ == 0) {
                continue;
            }

            X.at<float>(i, 0) = float(it2->pixX);
            X.at<float>(i, 1) = float(it2->pixY);
            X.at<float>(i, 2) = float(it2->pixZ);
            X.at<float>(i, 3) = 1;
            i++;
        }

        
        SVD::solveZ(X,coeff);
        float A = coeff.at<float>(0,0);
        float B = coeff.at<float>(1,0);
        float C = coeff.at<float>(2,0);
        float D = coeff.at<float>(3,0);
        
        //Interpolate information on superPixHole using A, B, C, D
        it2 = superPixHole.begin();
        for(; it2 != superPixHole.end(); it2++)
        {
            float predZ = (A * it2->pixX + B * it2->pixY + D)/( -1*C);
            if(predZ < 0 ) predZ = 0;
            if( predZ > NUM_FRAMES) predZ = 0;
            it2->pixZ = (int)round(predZ);
            depthMap.at<uchar>(it2->pixX, it2->pixY) = (it2->pixZ);
        }
        
        char ch;
    }

    return depthMap;
    
}


int main(int argc, char** argv)
{
	if( argc != 5)
	{
		cout << "Insufficent Arguments" << endl;
		cout << "Usage" << endl;
		cout << "./holeFilling <segmentation Img> <Reliable depth Img> <outDirectory> <numFrames>" << endl;
		return 1;
	}
	string seg_filename = argv[1];
	string reliable_depth_filename = argv[2];
	string outDirectory = argv[3];
	NUM_FRAMES = atoi(argv[4]);
	Mat segI, sparseI;

	segI = imread( seg_filename, 1);
	
	//depthI - reliable but sparse index information - 1:NUM_FRAME
	sparseI = imread( reliable_depth_filename, 0);
	imshow("Reliable Sparse Map", sparseI*10);

	
	//cout<< "Post Processing Sparse Map - Borrow\n";
	//sparseI = interpolation1( sparseI);
	//imshow("Post-processed Sparse Map - Borrow", sparseI*10);

	cout << "\nSegments processing\n";
	map<string, vector< Pixel > >segmentInfo;
	map<string, vector< Pixel > >segmentHole;
	countSegments( segI, sparseI, segmentInfo, segmentHole);

	cout << "\nInterpolate to get Dense Image - Plane fitting \n";
	Mat denseI = planeEstimation( segmentInfo, segmentHole, sparseI);
	
	cout << "\nPost processing Dense Map - Borrow\n";
	denseI = interpolation1( denseI);
	
	imshow("Final Dense Map", denseI*10);
	imwrite( outDirectory+"/denseMap.ppm", denseI);
	//waitKey(0);
}

