/***
 * Version 2
 * Laplacian stack calculated at orignal size and scaled down for all other levels. 
 * Variance Stack calculated at 4 different scales
 *
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

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


#define STACK_SIZE 3
//#define PATCH_SIZE 3

#define GAUSSIAN_BLUR 11
#define SIGMA 2

#define LAPLACIAN_SIZE 3

//#define SIGMA 2
using namespace std;
using namespace cv;

//int kernelSize[ STACK_SIZE ] ={ 11, 11, 15, 21 };
int PatchSize[ STACK_SIZE] = { 3, 3, 3, };
int Sigma[ STACK_SIZE] = {2,2,2};
int NUM_FRAMES = 25; 
int DEBUG_MODE = 1;



/*************** Read images with different focus setting and at different resolutions(scaled-down) and returns a scaledFocalStack ******************/
vector<vector<Mat>>reading(string srcDirectory)
{
  int numFrames = NUM_FRAMES;
  // scaledFocalStack is a vector of focal-stacks at multiple scales
  vector< vector<Mat> > scaledFocalStack(STACK_SIZE);
  vector<Mat> focalStack( NUM_FRAMES);
  vector<thread>  threads;
  mutex imgMutex;	

  //Reading images at original resolution and saving them in the focalStack vector
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
    vector<Mat> focalStack( NUM_FRAMES);
    threads.clear();
    scaleFactor = scaleFactor/2;
    vector<Mat> prev = scaledFocalStack[lenStack];


    //Scale images to a lower resolution and push them in the focalStack vector
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

/* Calculate laplacian for each of the image in focal stack at the original resolution and obtain a laplacian stack.
 * scaled down the laplacian stack at original resolution to obtain a scaledLaplacianStack ******************/
vector< vector<Mat> >getLaplacianStacks( vector<Mat> focalStack, string outDirectory)
{

  int numFrames = NUM_FRAMES;
  string directory1= outDirectory + "/laplacian/";
  struct stat st = {0};
  if ( stat( directory1.c_str(), &st) == -1)
  {
    mkdir( directory1.c_str(), 0700);
  }

  vector< vector<Mat> > scaledLaplacianStack(STACK_SIZE);
  vector<Mat> laplacianStack( NUM_FRAMES);

  vector<thread>  threads;
  mutex imgMutex;	
  //Apply laplacian operator to each image in the focal-stack at original resolution.
  for(int i=0;i < NUM_FRAMES; i++)
  {
    threads.push_back( thread([ =, &laplacianStack, &focalStack, &imgMutex]{

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
          imwrite( directory1+"Level_1_frame"+to_string(i+1)+".jpg",disp1);
          }   
          imgMutex.unlock();

          }));
  }
  for (int i = 0; i < threads.size(); i++)
    threads[i].join();

  float scaleFactor = 1.0;
  int level = 0;
  scaledLaplacianStack[level++] = laplacianStack;
  vector<Mat> original = laplacianStack;

  while( level < STACK_SIZE )
  {
    threads.clear();

    //vector<Mat> prev = scaledLaplacianStack[level-1];
    vector<Mat> laplacianStack( NUM_FRAMES);
    scaleFactor = scaleFactor/2;

    //Scale down the original laplacian stack to obtain scaledLaplacianStack at multi-scale.
    for( int i = 0; i < NUM_FRAMES; i++)
    {
      threads.push_back( thread([ =, &laplacianStack, &original, &imgMutex]{

            Mat image;
            resize( original[i], image, Size(), scaleFactor, scaleFactor, INTER_AREA);
            //resize( prev[i], image, Size(), 0.5 , 0.5, INTER_CUBIC);
            imgMutex.lock();
            laplacianStack[i]= image;
            imgMutex.unlock();
            }
            ));

    }
    for (int i = 0; i < threads.size(); i++)
      threads[i].join();

    scaledLaplacianStack[level] = laplacianStack;
    for( int i = 0; i < NUM_FRAMES; i++)
    {
      if(DEBUG_MODE)
      {
        Mat disp1,disp2;
        convertScaleAbs( laplacianStack[i], disp1);
        imwrite( directory1+"/Level_" + to_string(level+1) + "_frame_"+to_string(i+1)+".jpg",disp1);
      } 
    }	
    level++;
  }
  return scaledLaplacianStack;
}

/* focalStack is the Image Stack (at  a single scale)
 * Use it to get weightStack( varianceStack) 
 * Get weighted laplacian image and weighted infocus image.
 */
void getWeights( int level, string outDirectory, vector<Mat> focalStack, vector<Mat> laplacianStack,  Mat& weightedImg, Mat& weightedLap, vector<Mat>& weightStack )
{
  int numFrames = NUM_FRAMES;

  int patchSize = PatchSize[level-1];	

  cout << "Level: " << level << endl;

  char ch;
  vector<thread>  threads;

  mutex imgMutex;	

  Mat cumWeight;
  cumWeight = Mat::zeros(focalStack[0].size(), CV_64F);
  double start, end;
  start = cv::getTickCount();

  string directory1= outDirectory + "/variance";
  struct stat st = {0};
  if (stat( directory1.c_str(), &st) == -1)
  {
    mkdir( directory1.c_str(), 0700);
  }
  //Create weightStack from the images in focalStack
  for(int i = 0; i < numFrames; i++)
  {

    threads.push_back( thread([ =, &focalStack, &imgMutex, &laplacianStack, &weightStack]
          {
          Mat laplacianI, mu, mu2, varianceI;

          laplacianI = laplacianStack[i];
          //creating variance of laplacian image
          blur( laplacianI, mu, Size( 2*patchSize+1, 2*patchSize+1));
          blur( laplacianI.mul(laplacianI), mu2, Size(2*patchSize+1, 2*patchSize+1) );
          varianceI = mu2 - mu.mul(mu);
          varianceI = varianceI + 0.001;

          //Pushing unnormalized weight stack!!!	
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
  cout << "\t Creating Variance Stack: ";
  cout <<	(end-start)/getTickFrequency() << endl;


  start = cv::getTickCount();
  threads.clear();
  Mat avgI = Mat::zeros(cumWeight.size(), CV_64FC3);
  Mat lapI = Mat::zeros(cumWeight.size(), CV_64F);
  // Calculating partially focussed wighted average image (avgI) and laplacian of the average image (lapI)
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
      imwrite( directory1+"/Variance_stacklevel_" + to_string(level) + "_frame_"+to_string(i+1)+".jpg", draw);
    }
  }
  convertScaleAbs(avgI, weightedImg);
  weightedLap = lapI;
  end = cv::getTickCount();
  cout << "\t Creating weighted images: ";
  cout <<  (end-start)/getTickFrequency() << endl;
  return;
}

void getIndexMap( string outDirectory, Mat& indexMap, vector<Mat> &focalStack, vector<Mat> &laplacianStack, Mat avgI, Mat lapI, int level)
{

  cout << "Level: " << level << endl;

  int sigma = Sigma[level-1];	
  int patchSize = PatchSize[level-1];

  double start, end;
  vector<Mat> comparisonStack(NUM_FRAMES);

  vector<thread> threads;
  mutex imgMutex;
  char ch;

  indexMap = Mat::zeros(focalStack[0].size(), CV_8U);
  Mat indexImg = Mat::zeros(focalStack[0].size(), CV_8UC3);

  string directory1= outDirectory + "/indexMaps";
  struct stat st = {0};
  if (stat( directory1.c_str(), &st) == -1)
  {
    mkdir( directory1.c_str(), 0700);
  }


  start = cv:: getTickCount();
  Mat tmp1;

  GaussianBlur( lapI , tmp1 , Size( 2*patchSize+1, 2*patchSize+1), sigma, sigma);
  //Creating comparisonStack to determine the best focussed image in the focal-stack for each pixel.
  //This is done by measuring the similarity measure between lapI (average laplacian image) and each of the image in laplacain stack
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

          imgMutex.lock();
          comparisonStack[i] = disp3;
          imgMutex.unlock();
          }
          ));
  }

  for (int i = 0; i < threads.size(); i++)
    threads[i].join();

  //Use comparison stack, to find the maximum similariy response at each pixel over all images in focal-stack. 
  //This indicates the image-index for the best focussed image out of the focal-stack at a particular pixel position
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

      //indexMap.at<uchar>(y,x) = index*10;
      indexMap.at<uchar>(y,x) = index;

      //Redundant operations to create a colored sparse depth map
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

  cout << "\t Calculating Index Map and Index Image: " << (end-start)/getTickFrequency() << endl;
  imwrite( directory1+"/IndexImg_"+to_string(level)+".jpg", indexImg);
  imwrite( directory1+"/IndexMap_"+to_string(level)+".ppm", indexMap*10);
}

//Return a reliable sparse index-map from the multi-scale sparse index-maps
void getReliableIndexMap( string outDirectory ,vector<Mat> indexMaps, Mat reliableIndexMap)
{
  string directory1= outDirectory + "/indexMaps";
  struct stat st = {0};
  if (stat( directory1.c_str(), &st) == -1)
  {
    mkdir( directory1.c_str(), 0700);
  }

  //Resizing to the original resolution
  for(int i = 1; i < STACK_SIZE; i++ )
    resize( indexMaps[i], indexMaps[i], indexMaps[0].size());

  Mat minMap = Mat::zeros( indexMaps[0].size(), CV_8U);
  Mat maxMap = Mat::zeros( indexMaps[0].size(), CV_8U);
  reliableIndexMap = Mat::zeros( indexMaps[0].size(), CV_8U);
  //Idea: Ensure consistency of index across different multiscale sparse index-maps at each pixel-position
  for( int y = 0; y < indexMaps[0].rows; y++ )
  {
    for( int x = 0; x < indexMaps[0].cols; x++ )
    {
      int max = 0, min = 1000;
      for(int i = 0; i < STACK_SIZE; i++)
      {
        if( indexMaps[i].at<uchar>(y, x) < min )
          min = indexMaps[i].at<uchar>(y, x);
        if( indexMaps[i].at<uchar>(y, x) > max )
          max = indexMaps[i].at<uchar>(y, x);
      }
      //chk
      if( abs(min-max) > 3)
      {
        minMap.at<uchar>(y, x) = 0;
        maxMap.at<uchar>(y, x) = 0;
        reliableIndexMap.at<uchar>(y, x) = 0;
      }
      else
      {
        minMap.at<uchar>(y, x) = min;
        maxMap.at<uchar>(y, x) = max;
        reliableIndexMap.at<uchar>(y, x) = int(max+min)/2;
        //reliableIndexMap.at<uchar>(y, x) = (int(max+min)/20)*10;
      }
    }
  }
  //chk
  imwrite( outDirectory+"/reliableMap.ppm", reliableIndexMap);
  imwrite( directory1+"/minMap.jpg", minMap*10);
  imwrite( directory1+"/maxMap.jpg", maxMap*10);

}
int main( int argc, char** argv )
{
  if( argc != 5  )
  {
    cout << "Insufficient Arguments!! Enter filename \n"; 
    cout << "Usage:\n";
    cout << "directory without trailing /" << endl;
    cout << "./depthMap <input_directory> <output_directory> <debug_mode> <numImages>\n";

    return 1;
  }
  struct stat st = {0};
  if (stat( argv[2], &st) == -1)
  {
    mkdir( argv[2], 0700);
  }

  string srcDirectory = argv[1];
  string outDirectory = argv[2];
  DEBUG_MODE = atoi(argv[3]);
  NUM_FRAMES = atoi(argv[4]);



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

  //Step 1: Read images at multi-scale.
  cout << "Begin reading images!\n";
  start = cv::getTickCount();	

  scaledFocalStack = reading( srcDirectory);

  end = cv::getTickCount();
  cout << "Reading Images at " << STACK_SIZE << " levels: "  << (end-start)/getTickFrequency() << endl;

  //Step 2: Create laplacian stack at multi-scale ( Scaling the orignal stack )
  //						( Rather than recalculation )
  cout << "\n\nCreating Laplacian Stacks at Multiscale!\n"; 	
  start = cv::getTickCount();	

  scaledLaplacianStack = getLaplacianStacks( scaledFocalStack[0], outDirectory);

  end = cv::getTickCount();
  cout << "Laplacian stack created at " << STACK_SIZE << " levels: "  << (end-start)/getTickFrequency() << endl;


  //Step 3: Create Variance stack at multiscale and weighted images at multiple scales.

  cout << "\n\nCreating Variance Stacks at Multiscale!\n";

  //string directory = "./Results/Experiment_v2/weightedImages/";
  start = cv::getTickCount();
  for(int i=0; i < STACK_SIZE; i++)
  {
    Mat weightedImg, weightedLap;
    vector<Mat> varianceStack(NUM_FRAMES);

    getWeights( i+1, outDirectory,scaledFocalStack[i], scaledLaplacianStack[i], weightedImg, weightedLap, varianceStack);

    scaledWeightedImg.push_back( weightedImg);
    scaledWeightedLap.push_back( weightedLap);

    scaledVarianceStack.push_back( varianceStack);
    if( i == 0)
    {
      imwrite( outDirectory+"/averageImage"+ ".ppm", weightedImg);
    }
    if( DEBUG_MODE )
    {
      imwrite( outDirectory+"/averageImage_" + to_string(i) + ".jpg", weightedImg);
      imwrite( outDirectory+"/averageLaplacian_" + to_string(i) + ".jpg", weightedLap);
    }
  }
  end = cv::getTickCount();

  cout << "Variance Stack created at " << STACK_SIZE << " levels: " << (end-start)/getTickFrequency() << endl;

  //Step 4: Create Index map at multi-scale

  cout << "\n\nCreating Index Map at Multi-scale!\n";
  start = cv::getTickCount();
  for(int i=0; i < STACK_SIZE; i++)
  {
    Mat indexMap;
    getIndexMap( outDirectory, indexMap, scaledFocalStack[i], scaledLaplacianStack[i], scaledWeightedImg[i], scaledWeightedLap[i], i+1);
    scaleIndexMap.push_back(indexMap);
  }
  end = cv::getTickCount();
  cout << "IndexMap created at " << STACK_SIZE << " levels: " << (end-start)/getTickFrequency() << endl;

  //Step 5: Get Reliable Index Maps

  cout << "\n\nCreating Reliable Index Map!\n";
  start = cv::getTickCount();
  Mat reliableIndexMap;
  getReliableIndexMap( outDirectory, scaleIndexMap, reliableIndexMap);
  end = cv::getTickCount();
  cout << "Reliable IndexMap created: " << (end-start)/getTickFrequency() << endl;

  endFinal = cv::getTickCount();
  cout << "\n\nTotal time to get Reliable Map: "  << (endFinal-startFinal)/getTickFrequency() << endl;

  //Step 6: Get Dense Map - pass control to holeFilling.cpp
}

