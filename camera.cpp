#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

struct ColorDistribution {
    float data[8][8][8]; // l'histogramme
    int nb;              // le nombre d'échantillons
    
    ColorDistribution() { reset(); }
    ColorDistribution& operator=(const ColorDistribution& other) = default;
    
    // Met à zéro l'histogramme    
    void reset() {
        fill(&data[0][0][0], &data[0][0][0] + sizeof(data) / sizeof(float), 0.0f);
        nb = 0;
    }
    
    // Ajoute l'échantillon color à l'histogramme
    void add(Vec3b color) {
        int rBin = color[2] / 32;
        int gBin = color[1] / 32;
        int bBin = color[0] / 32;
        data[rBin][gBin][bBin]++;
        nb++;
    }
    
    // Indique qu'on a fini de mettre les échantillons
    void finished() {
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                for (int k = 0; k < 8; k++)
                    data[i][j][k] /= nb;
    }

    // Retourne la distance entre cet histogramme et l'histogramme other
    float distance(const ColorDistribution& other) const {
        float dist = 0.0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                    dist += abs(data[i][j][k] - other.data[i][j][k]);
                }
            }
        }
        return dist;
    }
};

ColorDistribution
getColorDistribution( Mat input, Point pt1, Point pt2 )
{
  ColorDistribution cd;
  for ( int y = pt1.y; y < pt2.y; y++ )
    for ( int x = pt1.x; x < pt2.x; x++ )
      cd.add( input.at<Vec3b>( y, x ) );
  cd.finished();
  return cd;
}

int main( int argc, char** argv )
{
  Mat img_input, img_seg, img_d_bgr, img_d_hsv, img_d_lab;
  VideoCapture* pCap = nullptr;
  const int width = 640;
  const int height= 480;
  const int size  = 50;
  // Ouvre la camera
  pCap = new VideoCapture( 0 );
  if( ! pCap->isOpened() ) {
    cout << "Couldn't open image / camera ";
    return 1;
  }
  // Force une camera 640x480 (pas trop grande).
  pCap->set( CAP_PROP_FRAME_WIDTH, 640 );
  pCap->set( CAP_PROP_FRAME_HEIGHT, 480 );
  (*pCap) >> img_input;
  if( img_input.empty() ) return 1; // probleme avec la camera
  Point pt1( width/2-size/2, height/2-size/2 );
  Point pt2( width/2+size/2, height/2+size/2 );
  namedWindow( "input", 1 );
  imshow( "input", img_input );
  bool freeze = false;
  while ( true )
    {
      char c = (char)waitKey(50); // attend 50ms -> 20 images/s
      if ( pCap != nullptr && ! freeze )
        (*pCap) >> img_input;     // récupère l'image de la caméra
      if ( c == 27 || c == 'q' )  // permet de quitter l'application
        break;
      if ( c == 'f' ) // permet de geler l'image
        freeze = ! freeze;
        if (c == 'v') {
            // Define the left and right parts of the frame
            Point left_pt1(0, 0), left_pt2(img_input.cols / 2, img_input.rows);
            Point right_pt1(img_input.cols / 2, 0), right_pt2(img_input.cols, img_input.rows);

            // Calculate color distributions for left and right halves
            ColorDistribution left_hist = getColorDistribution(img_input, left_pt1, left_pt2);
            ColorDistribution right_hist = getColorDistribution(img_input, right_pt1, right_pt2);

            // Calculate and display the distance between left and right histograms
            float dist = left_hist.distance(right_hist);
            cout << "Distance between left and right parts: " << dist << endl;
        }
      cv::rectangle( img_input, pt1, pt2, Scalar( { 255.0, 255.0, 255.0 } ), 1 );
      imshow( "input", img_input ); // affiche le flux video
    }
  return 0;
}