/**
 *
 * Version 0
 * All images processed at same level. Implemented by Changyin et al. 
 * Advantage: Much better solutions
 * Drawbacks: Processing at 6 scales takes 27 sec.
 */ 


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

#define GAUSSIAN_BLUR 11
#define SIGMA 2

#define LAPLACIAN_SIZE 3

#define NUM_FRAMES 25 
#define STACK_SIZE 6

int patchSize[STACK_SIZE] ={3,5,7,10,15,20};

int DEBUG_MODE = 1;
using namespace std;
using namespace cv;

/*************** Reading Images at single scale ******************/
vector<Mat> reading( string srcDirectory, int flagScale)
{
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
					if( flagScale)
						resize( image, image, Size(), 0.5 , 0.5, INTER_CUBIC);

					imgMutex.lock();
					focalStack[i]= image;
					imgMutex.unlock();

					}));
	}
	for (int i = 0; i < threads.size(); i++)
		threads[i].join();
	
	cout << "Size of Images in focal stack: " << focalStack[0].size() << endl;
	return focalStack;
}

/*************** Creating laplacian stack at single scale ******************/
vector<Mat> getLaplacianStack( vector<Mat> focalStack )
{
	vector<Mat> laplacianStack( NUM_FRAMES);
	vector<thread> threads;
	mutex imgMutex;
	string directory1= "./Results/Experiment2/Stacks/laplacian/";
	for( int i = 0; i < NUM_FRAMES; i++)
	{
		threads.push_back( thread([ =, &focalStack, &laplacianStack, &imgMutex]
					{
					Mat image,imageGray, laplacianI;
					image = focalStack[i];
					cvtColor( image, imageGray, CV_RGB2GRAY);
					GaussianBlur(imageGray, imageGray, Size(GAUSSIAN_BLUR, GAUSSIAN_BLUR), SIGMA, SIGMA);
					Laplacian( imageGray, laplacianI, CV_64F, LAPLACIAN_SIZE); //scale = 1, delta = 0		
					imgMutex.lock();
					laplacianStack[i] = laplacianI;
					if(DEBUG_MODE)
					{
						Mat disp1,disp2;
						convertScaleAbs( laplacianStack[i], disp1);
						imwrite( directory1+"Laplacian_stack_frame_"+to_string(i+1)+".jpg",disp1);
					}
					imgMutex.unlock();
					}
					));
	}
	for (int i = 0; i < threads.size(); i++)
		threads[i].join();

	return laplacianStack;
}

// focalStack is the Image Stack (at single scale) . Use it to get weightStack( varianceStack) and laplacianStack. As well as weighted laplacian img and weighted partially focused image.
void getWeights( int level, vector<Mat>& focalStack, vector<Mat>& laplacianStack, int patchSize, vector<Mat>& weightStack, Mat& weightedImg, Mat& weightedLap )
{
	
	vector<thread>  threads;
	mutex imgMutex;	
	
	Mat cumWeight;
	cumWeight = Mat::zeros(focalStack[0].size(), CV_64F);
	
	string directory1= "./Results/Experiment2/Stacks/variance/";
	double start, end;
	start = cv::getTickCount();	
	for(int i = 0; i < NUM_FRAMES; i++)
	{

		threads.push_back( thread([ =, &imgMutex, &laplacianStack, &weightStack]
					{
					Mat laplacianI, mu, mu2, varianceI;
					laplacianI = laplacianStack[i];

					//creating variance of laplacian image
					//GaussianBlur( laplacianI, mu, Size(2*patchSize+1, 2*patchSize+1),patchSize,patchSize);
					//GaussianBlur( laplacianI.mul(laplacianI), mu2, Size(2*patchSize+1, 2*patchSize+1),patchSize,patchSize);
					blur( laplacianI, mu, Size(2*patchSize+1, 2*patchSize+1));
					blur( laplacianI.mul(laplacianI), mu2, Size(2*patchSize+1, 2*patchSize+1));
					varianceI = mu2 - mu.mul(mu);
					varianceI = varianceI + 0.001;

					//Pushing unnormalized weights (variance values) on the weightStack
					imgMutex.lock();
					weightStack[i] = varianceI; 
					add(cumWeight, varianceI, cumWeight);
					imgMutex.unlock();
					}
		));
	}
	for (int i = 0; i < threads.size(); i++)
		threads[i].join();
	end = cv::getTickCount();
	cout << "Step 2a(Calculating variance stack): " <<  (end-start)/getTickFrequency() << endl;
	
	start = cv::getTickCount();
	threads.clear();
	Mat avgI = Mat::zeros(cumWeight.size(), CV_64FC3);
	Mat lapI = Mat::zeros(cumWeight.size(), CV_64F);
	
	// Calculating partially focussed wighted average image (avgI) and laplacian of that image (lapI)
	for(int i=0; i< focalStack.size(); i++)
	{
		threads.push_back( thread([ =, &imgMutex, &focalStack, &laplacianStack, &weightStack, &lapI, &avgI]
		{
		Mat tmp1;
		//weight stack normalized!!
		weightStack[i] = weightStack[i]/cumWeight; 
	
		focalStack[i].convertTo(tmp1, CV_64F);
		vector<Mat> channels;
		Mat mergedI;
		channels.push_back(weightStack[i]);
		channels.push_back(weightStack[i]);
		channels.push_back(weightStack[i]);
		merge(channels, mergedI);
		tmp1 = tmp1.mul(mergedI);
		avgI = avgI + tmp1;
		
		Mat tmp2 = laplacianStack[i].mul(weightStack[i]);	
		imgMutex.lock();
		lapI = lapI + tmp2;
		imgMutex.unlock();
		}
		));
	}

	for (int i = 0; i < threads.size(); i++)
		threads[i].join();
	if( DEBUG_MODE)
	{
		for( int i = 0; i < focalStack.size(); i++)
		{
			double minVal, maxVal;
			Mat draw, sigmaFiltered;
			convertScaleAbs(weightStack[i], sigmaFiltered);
			minMaxLoc( weightStack[i], &minVal, &maxVal);
			weightStack[i].convertTo(draw, CV_8U, 255.0/(maxVal));
			imwrite( directory1+"Variance_stack_level" + to_string(level) + "_frame_"+to_string(i+1)+".jpg", draw);
		}
	}

	convertScaleAbs(avgI, weightedImg);
	weightedImg = avgI;
	weightedLap = lapI;
	end = cv::getTickCount();
	cout << "Step 2b(Calculating weighted laplacian image): " <<  (end-start)/getTickFrequency() << endl;
	return;
}

void getIndexMap( Mat& indexMap, vector<Mat> &laplacianStack, Mat avgI, Mat lapI, int level, int patchRadius)
{
	double start, end;
	vector<Mat> comparisonStack(NUM_FRAMES);
	
	vector<thread> threads;
	mutex imgMutex;
	char ch;
        indexMap = Mat::zeros(laplacianStack[0].size(), CV_8U);
        Mat indexImg = Mat::zeros(laplacianStack[0].size(), CV_8UC3);
	//Mat blurredLapI;
	//blur( lapI, blurredLapI, Size( kernelSize, kernelSize));
	
	string directory1="./Results/Experiment2/Stacks/";
	string directory2="./Results/Experiment2/indexMaps/";
	
	start = cv:: getTickCount();
	Mat tmp1;
	GaussianBlur( lapI , tmp1, Size(2*patchRadius+1, 2*patchRadius+1), patchRadius, patchRadius);
	for(int i = 0; i < laplacianStack.size(); i++)
	{
		threads.push_back( thread([ =, &imgMutex, &laplacianStack, &comparisonStack]
		{
		Mat lapFrame = laplacianStack[i];
		Mat tmp,tmp2;
		GaussianBlur( lapI.mul(lapFrame), tmp , Size(2*patchRadius+1, 2*patchRadius+1), patchRadius, patchRadius);
		GaussianBlur( lapFrame, tmp2 , Size(2*patchRadius+1, 2*patchRadius+1), patchRadius, patchRadius);
		Mat differenceImg = abs( tmp - tmp1.mul(tmp2));

		Mat disp3;	
		convertScaleAbs( differenceImg, disp3);
		
		/*
		Mat disp1,disp2;
		convertScaleAbs( lapFrame, disp1);
		imwrite( directory1+"lap_stack_"+to_string(level)+"_frame"+to_string(i)+".jpg",disp1);	
		convertScaleAbs( correlationImg, disp2);
		imwrite( directory1+"corr_stack_"+to_string(level)+"_frame"+to_string(i)+".jpg",disp2);			
		disp3 = 255 - disp3;
		imwrite(directory1+"diff_stack_"+to_string(level)+"_frame"+to_string(i)+".jpg",disp3);	
		*/

		imgMutex.lock();
		comparisonStack[i] = disp3;
		imgMutex.unlock();
		}
		));
	}
	for (int i = 0; i < threads.size(); i++)
		threads[i].join();
	end = cv:: getTickCount();
	cout << "Step 3a( Creating comparison stack)" << (end-start)/getTickFrequency() << endl;
	
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
	cout << "Step 3b(Calculating Index Map and Index Image) " << (end-start)/getTickFrequency() << endl;
	imwrite( directory2+"IndexImg_"+to_string(level)+".jpg", indexImg);	
	imwrite( directory2+"IndexMap_"+to_string(level)+".jpg", indexMap);	
}
void getReliableIndexMap( vector<Mat> indexMaps, Mat reliableIndexMap)
{
	string directory2="./Results/Experiment2/indexMaps/";

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
}
int main( int argc, char** argv )
{
	if( argc < 3  )
	{
		cout << "Insufficient Arguments!! Try: ./depthMap stairs11/ 0 \n"; 
   		cout << "Usage:\n";
		cout << "./depthMap <dataset_directory> <scalingFlag>\n";

		return 1;
	}
	
	string srcDirectory = argv[1];
	double start, end;
	double startFinal, endFinal;
	
	startFinal = cv::getTickCount();	
	// Step - 1: Reading Images at Single-scale in focalStack .
	
	vector<Mat> focalStack;
	int scaleFlag = atoi(argv[2]);
	start = cv::getTickCount();	
	focalStack = reading( srcDirectory, scaleFlag);
	end = cv::getTickCount();
	cout << "\tReading Images: " << (end-start)/getTickFrequency() << endl;
	
	
	char ch;
	string directory = "./Results/Experiment2/weightedImages/";
	string directory1 = "./Results/Experiment2/laplacianStack/";
	
	// Step - 2a: Creating Single-scale laplacian stack.
	vector<Mat> laplacianStack;
	start = cv::getTickCount();
	laplacianStack = getLaplacianStack( focalStack);
	end = cv::getTickCount();
	//Routine to print the images in laplacian stack!!
	for( int i=0; i< NUM_FRAMES; i++)
	{
		double minVal, maxVal;
		Mat draw;
		minMaxLoc(laplacianStack[i], &minVal, &maxVal);
		laplacianStack[i].convertTo(draw, CV_8U, 255.0/(maxVal));
		imwrite( directory1+"lap_"+to_string(i) + ".jpg", draw);
	}
	cout << "\tLaplacian Stack: " << (end-start)/getTickFrequency() << endl;

	// Step - 2b:  Creating Multi-scale weight stack aka (variance stack).
	// and Multiscale weighted Images
	// and Multiscale weighted Laplacian Images
	vector< vector<Mat> >  scaledVarianceStack; 
	vector<Mat> scaledWeightedImg;
	vector<Mat> scaledWeightedLap;
	vector<Mat> scaleIndexMap;
	start = cv::getTickCount();
	for(int i = 0; i < STACK_SIZE; i++)
	{
		Mat weightedImg, weightedLap;
		vector<Mat> varianceStack(NUM_FRAMES);
		
		getWeights( i+1, focalStack, laplacianStack, patchSize[i], varianceStack, weightedImg, weightedLap );
		
		scaledWeightedImg.push_back( weightedImg);
		scaledWeightedLap.push_back( weightedLap);
		
		scaledVarianceStack.push_back( varianceStack);
		
		imwrite( directory+"weightedImg_"+to_string(patchSize[i]) + ".jpg", weightedImg);
		double minVal, maxVal;
		Mat draw;
		minMaxLoc( weightedLap, &minVal, &maxVal);
		weightedLap.convertTo(draw, CV_8U, 255.0/(maxVal-minVal),-255.0/minVal);
		imwrite( directory+"weightedLaplacian_"+to_string( patchSize[i])+ ".jpg", weightedLap);
	}
	end = cv::getTickCount();
	cout << "\tCummalative Stacks " << (end-start)/getTickFrequency() << endl;
	
	// Step 3 - get index maps at 3 scales
	start = cv::getTickCount();
	for(int i=0; i < STACK_SIZE; i++)
	{
		Mat indexMap;
		getIndexMap( indexMap, laplacianStack, scaledWeightedImg[i], scaledWeightedLap[i], i, patchSize[i]);
		scaleIndexMap.push_back(indexMap);
	}
	end = cv::getTickCount();
	cout << "\tGet IndexMap at " << STACK_SIZE << " levels: "  << (end-start)/getTickFrequency() << endl;

	//Step 4 - get reliable index maps
	start = cv::getTickCount();
	Mat reliableIndexMap;
	getReliableIndexMap( scaleIndexMap, reliableIndexMap);
	end = cv::getTickCount();
	cout << "\tGet Reliable indexMap " << (end-start)/getTickFrequency() << endl;
	
	endFinal = cv::getTickCount();
	cout << "\tTotal time: "  << (endFinal-startFinal)/getTickFrequency() << endl;
	//Step 5 - get dense index maps
}

