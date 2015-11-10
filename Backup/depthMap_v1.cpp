/***
 * Version 1
 * Laplacian stack calculated at 4 different scales
 * Variance Stack calculated at 4 different scales
 *
 * Constraint: Laplacian stacks for downsampled images at level 3 ( downsampled by 1/8)
 * 						 	and level 4( downsampled by 1/16) messed up.
 * 	      Identical laplacian images for all the images in the focal-stack at smaller levels. 
 * Advantage: 5 secs
 ***/


#include<iostream>
#include<string>
#include <cv.h>
#include <highgui.h>
#include<ctime>

#include<cstdlib>
#include<algorithm>
#include<iterator>
#include<thread>
#include<mutex>

#define STACK_SIZE 4
//#define PATCH_SIZE 3
#define LAPLACIAN_SIZE 3
//#define SIGMA 2
using namespace std;
using namespace cv;

int kernelSize[ STACK_SIZE ] ={ 11, 11, 15, 21 };
int Sigma[ STACK_SIZE ] ={ 2, 2, 2, 2};
int PatchSize[ STACK_SIZE] = { 3, 3, 3, 3};

int NUM_FRAMES = 25; 
int DEBUG_MODE = 1;



/*************** Reading Images at three scales (STACK_SIZE). At each level 25 frames (NUM_FRAMES) ******************/
vector< vector<Mat> >reading( string srcDirectory)
{
	int numFrames = NUM_FRAMES;
	vector< vector<Mat> > scaledFocalStack(STACK_SIZE);
	vector<Mat> focalStack( NUM_FRAMES);
	vector<thread>  threads;
	mutex imgMutex;	
	
	for(int i=0;i < NUM_FRAMES; i++)
	{
		threads.push_back( thread([ =, &focalStack, &imgMutex]{
					Mat image;
					string imgname;
					imgname = srcDirectory + "/frame" + to_string(i+1) + ".jpg";
					image = imread(imgname, 1);

					imgMutex.lock();
					focalStack[i]= image;
					imgMutex.unlock();

					}));
	}
	for (int i = 0; i < threads.size(); i++)
		threads[i].join();
	
	float scaleFactor = 1;
	int lenStack = 0;
	scaledFocalStack[lenStack] = focalStack;
	cout << "\tSize of Orignal Images: " << focalStack[0].size() << endl;
	while( lenStack < STACK_SIZE - 1 )
	{
		//focalStack.clear();
		vector<Mat> focalStack( NUM_FRAMES);
		threads.clear();
		scaleFactor = scaleFactor/2;
		vector<Mat> prev = scaledFocalStack[lenStack];
		

		for( int i = 0; i < NUM_FRAMES; i++)
		{
			threads.push_back( thread([ =, &focalStack, &prev, &imgMutex]
						{
						Mat image;
						resize( prev[i], image, Size(), 0.5 , 0.5, INTER_CUBIC);
						imgMutex.lock();
						focalStack[i]= image;
						imgMutex.unlock();
						}
						));
				

		}
		for (int i = 0; i < threads.size(); i++)
			threads[i].join();
		lenStack++;
		scaledFocalStack[lenStack] = focalStack;
	}
	return scaledFocalStack;
}

// focalStack is the Image Stack (at single scale)
// Use it to get weightStack( varianceStack) and laplacianStack.
// Get weighted laplacian image and weighted infocus image.

void createStack( int level, vector<Mat> focalStack, Mat& weightedImg, Mat& weightedLap, vector<Mat>& weightStack, vector<Mat>& laplacianStack )
{
	int numFrames = NUM_FRAMES;
	
	int patchSize = PatchSize[level-1];	
	int gaussKernel = kernelSize[level-1];
	int sigma = Sigma[level-1];
	
	cout << "Level: " << level << endl;
	
	char ch;
	vector<thread>  threads;

	mutex imgMutex;	
	
	Mat cumWeight;
	cumWeight = Mat::zeros(focalStack[0].size(), CV_64F);
	double start, end;
	start = cv::getTickCount();	
	string directory1= "./Results/Experiment1/Stacks/laplacian/";
	string directory2= "./Results/Experiment1/Stacks/variance/";
	for(int i = 0; i < numFrames; i++)
	{

		threads.push_back( thread([ =, &focalStack, &imgMutex, &laplacianStack, &weightStack]
				{
					Mat image, imageGray, laplacianI, mu, mu2, varianceI;
					image = focalStack[i];

					//creating laplacian image
					cvtColor( image, imageGray, CV_RGB2GRAY );
					GaussianBlur( imageGray, imageGray, Size(gaussKernel, gaussKernel), sigma, sigma);
					Laplacian( imageGray, laplacianI, CV_64F, LAPLACIAN_SIZE ); //scale = 1, delta = 0		

					//creating variance of laplacian image
					blur( laplacianI, mu, Size( 2*patchSize+1, 2*patchSize+1));
					blur( laplacianI.mul(laplacianI), mu2, Size(2*patchSize+1, 2*patchSize+1) );
					varianceI = mu2 - mu.mul(mu);
					varianceI = varianceI + 0.001;
					
					//Pushing unnormalized weight stack!!!	
					imgMutex.lock();
					weightStack[i] = varianceI; 
					laplacianStack[i] = laplacianI;
					add(cumWeight, varianceI, cumWeight);
					imgMutex.unlock();
				}
		));
	}
	for (int i = 0; i < threads.size(); i++)
		threads[i].join();
	end = cv::getTickCount();
	cout << "\t Creating Laplacian Stack and Variance Stack: ";
      	cout <<	(end-start)/getTickFrequency() << endl;
	
	if( DEBUG_MODE)
	{
		for(int i=0; i <numFrames; i++)
		{
			Mat disp1;
			convertScaleAbs( laplacianStack[i], disp1);
			imwrite( directory1+"Laplacian_stack_level_" + to_string(level) + "frame_"+to_string(i+1)+".jpg",disp1);

		}
	}
	

	start = cv::getTickCount();
	threads.clear();
	Mat avgI = Mat::zeros(cumWeight.size(), CV_64FC3);
	Mat lapI = Mat::zeros(cumWeight.size(), CV_64F);
	// Calculating partially focussed wighted average image (avgI) and laplacian of that image (lapI)
	for(int i=0; i< focalStack.size(); i++)
	{
		threads.push_back( thread([ =, &imgMutex, &focalStack, &laplacianStack, &weightStack, &lapI, &avgI]
		{
		//weight stack normalized!!
		weightStack[i] = weightStack[i]/cumWeight; 
		
		Mat tmp1;
		focalStack[i].convertTo(tmp1, CV_64F);
		
		vector<Mat> channels;
		Mat mergedI;
		channels.push_back(weightStack[i]);
		channels.push_back(weightStack[i]);
		channels.push_back(weightStack[i]);
		merge(channels, mergedI);
		tmp1 = tmp1.mul(mergedI);
		Mat tmp2 = laplacianStack[i].mul(weightStack[i]);	
		
		imgMutex.lock();
		avgI = avgI + tmp1;
		lapI = lapI + tmp2;
		imgMutex.unlock();
		
		}
		));
	}
	for (int i = 0; i < threads.size(); i++)
		threads[i].join();
	
	if(DEBUG_MODE)
	{
		for(int i=0; i < numFrames; i++)
		{
			double minVal, maxVal;
			Mat draw, sigmaFiltered;
			convertScaleAbs( weightStack[i], sigmaFiltered);
			minMaxLoc(weightStack[i], &minVal, &maxVal);
			weightStack[i].convertTo(draw, CV_8U, 255.0/(maxVal));
			imwrite( directory2+"Variance_stack_level" + to_string(level) + "_frame_"+to_string(i+1)+".jpg", draw);
		}
	}
	convertScaleAbs(avgI, weightedImg);
	weightedLap = lapI;
	end = cv::getTickCount();
	cout << "\t Creating weighted images: ";
	cout <<  (end-start)/getTickFrequency() << endl;
	return;
}

void getIndexMap( Mat& indexMap, vector<Mat> &focalStack, vector<Mat> &laplacianStack, Mat avgI, Mat lapI, int level)
{
	
	int sigma = Sigma[level-1];	
	int patchSize = PatchSize[level-1];
	
	double start, end;
	vector<Mat> comparisonStack(NUM_FRAMES);
	
	vector<thread> threads;
	mutex imgMutex;
	char ch;
        
	indexMap = Mat::zeros(focalStack[0].size(), CV_8U);
        Mat indexImg = Mat::zeros(focalStack[0].size(), CV_8UC3);
	
	string directory1="./Results/Experiment1/Stacks/";
	string directory2="./Results/Experiment1/indexMaps/";
	
	start = cv:: getTickCount();
	Mat tmp1;
	
	GaussianBlur( lapI , tmp1 , Size( 2*patchSize+1, 2*patchSize+1), sigma, sigma);
	for(int i = 0; i < focalStack.size(); i++)
	{
		threads.push_back( thread([ =, &imgMutex, &laplacianStack, &comparisonStack]
		{
		Mat lapFrame = laplacianStack[i];
		Mat tmp,tmp2;
		GaussianBlur( lapI.mul(lapFrame), tmp , Size( 2*patchSize+1, 2*patchSize+1), sigma, sigma);
		GaussianBlur( lapFrame, tmp2 , Size( 2*patchSize+1, 2*patchSize+1), sigma, sigma);
		Mat differenceImg = abs( tmp - tmp1.mul(tmp2));

		Mat disp3;	
		convertScaleAbs( differenceImg, disp3);

		/*
		Mat disp1,disp2;
		convertScaleAbs( lapFrame, disp1);
		imwrite( directory1+"lap_stack_"+to_string(level)+"_frame"+to_string(i)+".jpg",disp1);	
		convertScaleAbs( correlationImg, disp2);
		imwrite( directory1+"corr_stack_"+to_string(level)+"_frame"+to_string(i)+".jpg",disp2);	
		*/	
		
		//imwrite(directory1+"diff_stack_"+to_string(level)+"_frame"+to_string(i)+".jpg",disp3);	
		imgMutex.lock();
		comparisonStack[i] = disp3;
		imgMutex.unlock();
		}
		));
	}
	for (int i = 0; i < threads.size(); i++)
		threads[i].join();
	end = cv:: getTickCount();
	cout << "Step 3a(parallel load)" << (end-start)/getTickFrequency() << endl;
	
	start = cv:: getTickCount();
	// Calculating Index map and Index image
	for( int y = 0; y < indexMap.rows; y++ )
	{
		for( int x = 0; x < indexMap.cols; x++ )
		{
			double max = comparisonStack[0].at<uchar>(y,x);
			int index = 0;
			for( int i = 0; i < comparisonStack.size(); i=i+1 )
			{
				if( max < comparisonStack[i].at<uchar>(y, x) )
				{
					index = i+1;
					max = comparisonStack[i].at<uchar>(y, x);
				}
			}
			indexMap.at<uchar>(y,x) = ( index)*10;
			if( index >= 22 && index <= 25)
			{
				indexImg.at<Vec3b>(y,x)[0] = 255;
				indexImg.at<Vec3b>(y,x)[1] = 255;
				indexImg.at<Vec3b>(y,x)[2] = 255;
			}
			else if( index >= 18 && index <= 21)
				indexImg.at<Vec3b>(y,x)[0] = 255;
			else if( index >= 11 && index <= 17)
				indexImg.at<Vec3b>(y,x)[1] = 255;
			else if( index >= 5 && index <= 10)
				indexImg.at<Vec3b>(y,x)[2] = 255;
			else if( index >= 1 && index <=4 )
			{
				indexImg.at<Vec3b>(y,x)[1] = 255;
				indexImg.at<Vec3b>(y,x)[2] = 255;

			}
			else
			{	
				indexImg.at<Vec3b>(y,x)[0] = 0;
				indexImg.at<Vec3b>(y,x)[1] = 0;
				indexImg.at<Vec3b>(y,x)[2] = 0;
			}
		}
	}
	end = cv::getTickCount();
	cout << "Step 3b(sequential load) " << (end-start)/getTickFrequency() << endl;
	imwrite( directory2+"IndexImg_"+to_string(level)+".jpg", indexImg);	
	imwrite( directory2+"IndexMap_"+to_string(level)+".jpg", indexMap);	
}
void getReliableIndexMap( vector<Mat> indexMaps, Mat reliableIndexMap)
{
	string directory2="./Results/Experiment1/indexMaps/";
	for(int i = 1; i < STACK_SIZE; i++ )	
		resize( indexMaps[i], indexMaps[i], indexMaps[0].size());
        Mat minMap = Mat::zeros( indexMaps[0].size(), CV_8U);
        Mat maxMap = Mat::zeros( indexMaps[0].size(), CV_8U);
        reliableIndexMap = Mat::zeros( indexMaps[0].size(), CV_8U);

	for( int y = 0; y < indexMaps[0].rows; y++ )
	{
		for( int x = 0; x < indexMaps[0].cols; x++ )
		{
			int max = 0, min = 260;
			for(int i = 0; i < STACK_SIZE; i++)
			{
				if( indexMaps[i].at<uchar>(y, x) < min )
					min = indexMaps[i].at<uchar>(y, x);
				if( indexMaps[i].at<uchar>(y, x) > max )
					max = indexMaps[i].at<uchar>(y, x);
			}
			if( abs(min-max) > 30)
			{
				minMap.at<uchar>(y, x) = 0;
				maxMap.at<uchar>(y, x) = 0;
				reliableIndexMap.at<uchar>(y, x) = 0;
			}
			else
			{
				minMap.at<uchar>(y, x) = min;
				maxMap.at<uchar>(y, x) = max;
				reliableIndexMap.at<uchar>(y, x) = (int(max+min)/20)*10;
			}
		}
	}
	imwrite( directory2+"ReliableMap.jpg", reliableIndexMap);
	imwrite( directory2+"minMap.jpg", minMap);
	imwrite( directory2+"maxMap.jpg", maxMap);

	//cout << "Size of Images: " << indexMaps[0].size() << " " << indexMaps[1].size() << endl;
	//destroyWindow("Mask");

}
int main( int argc, char** argv )
{
	if( argc < 4  )
	{
		cout << "Insufficient Arguments!! Enter filename \n"; 
   		cout << "Usage:\n";
		cout << "./depthMap <dataset_directory> <output_directory_tag> <numImages>\n";

		return 1;
	}
	
	string srcDirectory = argv[1];
	string outDirectory = argv[2];
	NUM_FRAMES = atoi(argv[3]);
	cout << "Number of frames: " << NUM_FRAMES << endl;	
	//Timer variables
	double start, end;
	double startFinal, endFinal;
	
	vector< vector<Mat> >  scaledFocalStack; 
	vector< vector<Mat> >  scaledLaplacianStack; 
	vector< vector<Mat> >  scaledVarianceStack; 
	
	vector<Mat> scaledWeightedImg;
	vector<Mat> scaledWeightedLap;
	vector<Mat> scaleIndexMap;
	
	
	startFinal = cv::getTickCount();	
	

	cout << "Begin reading images!\n"; 	
	start = cv::getTickCount();	
	
	scaledFocalStack = reading( srcDirectory);
	
	end = cv::getTickCount();
	cout << "Reading Images at " << STACK_SIZE << " levels: "  << (end-start)/getTickFrequency() << endl;
	
	char ch;
	string directory = "./Results/Experiment1/weightedImages/";
	
	// Step - 2: create laplacian and weight(variance) stacks at multiple scales.
	start = cv::getTickCount();
	for(int i=0; i < STACK_SIZE; i++)
	{
		Mat weightedImg, weightedLap;
		vector<Mat> laplacianStack(NUM_FRAMES);
		vector<Mat> varianceStack(NUM_FRAMES);
			
		createStack( i+1, scaledFocalStack[i],  weightedImg, weightedLap, varianceStack, laplacianStack );
		
		scaledWeightedImg.push_back( weightedImg);
		scaledWeightedLap.push_back( weightedLap);
		
		scaledLaplacianStack.push_back( laplacianStack);
		scaledVarianceStack.push_back( varianceStack);
		
		imwrite( directory+"weightedImg_"+to_string(i) + ".jpg", weightedImg);
		imwrite( directory+"weightedLaplacian_"+to_string(i)+ ".jpg", weightedLap);
	}
	end = cv::getTickCount();
	cout << "Get Laplacian and Variance  Stacks: " << (end-start)/getTickFrequency() << endl;
	
	/*
	// Step 3 - get index maps at 4 scales
	start = cv::getTickCount();
	for(int i=0; i < STACK_SIZE; i++)
	{
		Mat indexMap;
		getIndexMap( indexMap, scaledFocalStack[i], scaledLaplacianStack[i], scaledWeightedImg[i], scaledWeightedLap[i], i);
		scaleIndexMap.push_back(indexMap);
	}
	end = cv::getTickCount();
	cout << "\t Get IndexMap: "  << (end-start)/getTickFrequency() << endl;

	Mat reliableIndexMap;
	getReliableIndexMap( scaleIndexMap, reliableIndexMap);
	endFinal = cv::getTickCount();
	
	cout << "Total time: "  << (endFinal-startFinal)/getTickFrequency() << endl;
	//Step 4 - get reliable index maps
	//Step 5 - get dense index maps
	*/
}

