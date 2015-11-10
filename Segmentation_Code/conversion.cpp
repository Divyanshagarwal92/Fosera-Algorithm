#include<cv.h>
#include<highgui.h>
#include<iostream>
#include<cstring>

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
	if( argc != 4)
	{
		cout << " ./conversion <filename> <format_from> <format_to>" << endl;
		return -1;
	}

	string filename = argv[1];
	string from = argv[2];
	string to = argv[3];
	
	Mat image;
	image = imread( filename + "." + from, CV_LOAD_IMAGE_COLOR);   // Read the file

	if(! image.data )                              // Check for invalid input
	{
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}

	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	//imshow( "Display window", image );                   // Show our image inside it.
	//waitKey(0);                                          // Wait for a keystroke in the window
	imwrite( filename + "."+ to, image);
	
	return 0;
}
