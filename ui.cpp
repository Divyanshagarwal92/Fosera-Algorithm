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

using namespace std;
using namespace cv;

int NUM_FRAMES;

Mat averageImage;
Mat denseMap;
Mat segI;
vector<Mat> focalStack;

vector<Mat> reading( string srcDirectory )
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

                                        imgMutex.lock();
                                        focalStack[i]= image;
                                        imgMutex.unlock();

                                        }));
        }
        for (int i = 0; i < threads.size(); i++)
                threads[i].join();

        cout << "Size of Orignal Images: " << focalStack[0].size() << endl;
	cout << "Number of images: " << focalStack.size() << endl;
	return focalStack;
}


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	int index = denseMap.at<Vec3b>(y,x)[0];
	
	if  ( event == EVENT_RBUTTONDOWN )
	{
		imshow("Refocussing App", denseMap*10);
		cout << "Left Click: Row "<< y << " Column:  " << x  << " Index: "<< index << endl;
	}
	else if  ( event == EVENT_LBUTTONDOWN )
	{
		if(index == 0 || index > NUM_FRAMES )
			imshow("Refocussing App", averageImage);
		else
			imshow("Refocussing App", focalStack[index-1]);
		cout << "Right Click: Row "<< y << " Column:  " << x  << " Index: "<< index << endl;
	}
	else if  ( event == EVENT_MBUTTONDOWN )
	{
		imshow("Refocussing App", segI);
	}
	/*
	else if ( event == EVENT_MOUSEMOVE )
	{
		cout << "Hover position(x,y) (" << x << ", " << y << ")" << " Intensity: " << intensity << endl;

	}
	*/
}
int main( int argc, char** argv )
{
        if( argc != 4 )
        {
                cout << "Insufficient Arguments!!\n";
                cout << "Usage:\n";
        	cout << "directory without trailing /" << endl;
                cout << "./ui <input_directory> <output_directory> <numImages>\n";

                return 1;
	}

	string srcDirectory = argv[1];
	string outDirectory = argv[2];
	NUM_FRAMES = atoi(argv[3]);


	focalStack = reading(srcDirectory);

	string imgname = outDirectory + "/denseMap.ppm";
	denseMap = imread( imgname, 1);
	
	imgname = outDirectory + "/averageImage.ppm";
	averageImage = imread( imgname, 1);
	
	imgname = outDirectory + "/averageImage-seg.ppm";
	segI = imread( imgname, 1);
	if ( denseMap.empty() ) 
	{ 
		cout << "Error loading the image" << endl;
		return -1; 
	}

	namedWindow("Refocussing App", 1);
	moveWindow("Refocussing App", 400,0);
	setMouseCallback("Refocussing App", CallBackFunc, NULL);
	imshow("Refocussing App", denseMap*10);
	waitKey(0);
	return 0;
}
