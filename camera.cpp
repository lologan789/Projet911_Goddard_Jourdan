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

// Calculate the minimum distance between histogram h and the histograms in hists
float minDistance(const ColorDistribution& h, const std::vector<ColorDistribution>& hists) {
    float minDist = FLT_MAX;
    for (const auto& hist : hists) {
        float dist = h.distance(hist);
        if (dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}


Mat recoObject(Mat input,
               const std::vector<std::vector<ColorDistribution>>& all_col_hists,
               const std::vector<Vec3b>& colors,
               const int bloc) {
    Mat output = input.clone();
    int height = input.rows;
    int width = input.cols;

    for (int y = 0; y <= height - bloc; y += bloc) {
        for (int x = 0; x <= width - bloc; x += bloc) {
            // Calculate the color histogram for the current block
            ColorDistribution h = getColorDistribution(input, Point(x, y), Point(x + bloc, y + bloc));

            // Find the closest histogram set (background or one of the objects)
            int closestIndex = -1;
            float minDist = std::numeric_limits<float>::max();

            for (size_t i = 0; i < all_col_hists.size(); ++i) {
                float dist = minDistance(h, all_col_hists[i]);
                if (dist < minDist) {
                    minDist = dist;
                    closestIndex = i;
                }
            }

            // Use the color associated with the closest match to label the block
            if (closestIndex >= 0) { // Ensure we have a valid color index
                Vec3b color = colors[closestIndex];
                rectangle(output, Point(x, y), Point(x + bloc, y + bloc), color, FILLED);
            }
        }
    }

    return output;
}

bool isSimilarToBackground(const Vec3b& color, const vector<Vec3b>& existing_colors) {
    for (const auto& existing_color : existing_colors) {
        if (cv::norm(color - existing_color) < 50) {  // Adjust threshold as necessary
            return true;
        }
    }
    return false;
}

// Regroupement des blocs 4x4 et étiquetage basé sur la majorité des couleurs
void groupBlocksAndLabel(Mat& img_input, const std::vector<std::vector<ColorDistribution>>& all_col_hists,
                         const std::vector<Vec3b>& colors, const int bloc, Mat& output) {
    int height = img_input.rows;
    int width = img_input.cols;

    for (int y = 0; y <= height - bloc * 4; y += bloc * 4) {
        for (int x = 0; x <= width - bloc * 4; x += bloc * 4) {
            // Calcul des histogrammes de couleur pour le groupe de blocs 4x4
            std::vector<ColorDistribution> groupHists;
            for (int dy = 0; dy < 4; ++dy) {
                for (int dx = 0; dx < 4; ++dx) {
                    Point block_pt1(x + dx * bloc, y + dy * bloc);
                    Point block_pt2(x + (dx + 1) * bloc, y + (dy + 1) * bloc);
                    ColorDistribution block_hist = getColorDistribution(img_input, block_pt1, block_pt2);
                    groupHists.push_back(block_hist);
                }
            }

            // Trouver le label majoritaire dans le groupe de blocs
            int label = -1;
            float minDist = FLT_MAX;
            for (size_t i = 0; i < all_col_hists.size(); ++i) {
                float dist = minDistance(groupHists[0], all_col_hists[i]);
                if (dist < minDist) {
                    minDist = dist;
                    label = i;
                }
            }

            // Attribuer la couleur du label au groupe de blocs
            if (label >= 0) {
                Vec3b groupColor = colors[label];
                rectangle(output, Point(x, y), Point(x + bloc * 4, y + bloc * 4), groupColor, FILLED);
            }
        }
    }
}

void applyWatershedSegmentation(Mat& img_input, Mat& output) {
    Mat markers = Mat::zeros(img_input.size(), CV_32S);
    // Marquage du fond (0) et des objets (1, 2, etc.)
    // Vous pouvez ajuster les labels comme nécessaire
    for (int i = 0; i < img_input.rows; ++i) {
        for (int j = 0; j < img_input.cols; ++j) {
            if (output.at<Vec3b>(i, j) == Vec3b(255, 255, 255)) {  // Fond (par exemple)
                markers.at<int>(i, j) = 0;  // Fond
            } else {
                markers.at<int>(i, j) = 1;  // Objet
            }
        }
    }

    // Appliquer l'algorithme watershed
    watershed(img_input, markers);

    // Appliquer les résultats de watershed à l'image de sortie
    Mat watershed_result = Mat::zeros(img_input.size(), CV_8UC3);
    img_input.copyTo(watershed_result, markers == 1);

    // Ajouter la segmentation à l'image de sortie
    output = watershed_result;
}



int main( int argc, char** argv )
{
  Mat img_input, img_seg, img_d_bgr, img_d_hsv, img_d_lab;
  VideoCapture* pCap = nullptr;
  const int width = 640;
  const int height= 480;
  const int size  = 50;
  const int bbloc = 128;
  const int reco_bloc = 16;

  // histogrammes du fond
  std::vector<ColorDistribution> col_hists;
  // histogrammes de l'objet
  std::vector<ColorDistribution> col_hists_object;
  std::vector<std::vector<ColorDistribution>> all_col_hists = {col_hists, col_hists_object};
  vector<Vec3b> colors = { Vec3b(0, 0, 0), Vec3b(0, 0, 255) };

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
  bool reco = false;
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
        if (c == 'b') { // Calculate color distributions for background
            col_hists.clear(); // Clear previous background histograms
            for (int y = 0; y <= height - bbloc; y += bbloc) {
                for (int x = 0; x <= width - bbloc; x += bbloc) {
                    Point block_pt1(x, y);
                    Point block_pt2(x + bbloc, y + bbloc);
                    ColorDistribution block_hist = getColorDistribution(img_input, block_pt1, block_pt2);
                    col_hists.push_back(block_hist);
                    all_col_hists[0].push_back(block_hist); // Ensure it’s stored in all_col_hists
                }
            }
            int nb_hists_background = col_hists.size();
            cout << "Number of background histograms: " << nb_hists_background << endl;
        }
        if (c == 'a') { // Calculate and store histogram for the object area
            col_hists_object.clear(); // Clear previous object histograms
            ColorDistribution object_hist = getColorDistribution(img_input, pt1, pt2);
            col_hists_object.push_back(object_hist);
            all_col_hists[0] = col_hists_object; // Ensure it’s stored in all_col_hists
            cout << "Object histogram added. Total object histograms: " << col_hists_object.size() << endl;
        }
        if (c == 'n') { // Add a new object
            std::vector<ColorDistribution> new_object_hist;
            ColorDistribution new_hist = getColorDistribution(img_input, pt1, pt2);
            new_object_hist.push_back(new_hist);
            all_col_hists.push_back(new_object_hist);
            
            // Define a new color for this object
            Vec3b new_color(rand() % 256, rand() % 256, rand() % 256);
            while (isSimilarToBackground(new_color, colors)) {  // Assuming you have a function that checks color similarity
                new_color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
            }
            colors.push_back(new_color);

            
            cout << "New object added. Total objects: " << all_col_hists.size() - 1 << endl;
        }
        if (c == 'r') {
            reco = !reco;
        }
        if (c == 'c') { // Reset all histograms
            col_hists.clear();
            col_hists_object.clear();
            all_col_hists = {col_hists, col_hists_object};
            colors = { Vec3b(0, 0, 0), Vec3b(0, 0, 255) };
            cout << "Histograms and objects reset." << endl;
        }
        Mat output = img_input;
        if (reco && !col_hists.empty()) {
            Mat gray;
            cvtColor(img_input, gray, COLOR_BGR2GRAY);
            Mat reco_img = recoObject(img_input, all_col_hists, colors, reco_bloc);
            cvtColor(gray, img_input, COLOR_GRAY2BGR);
            output = 0.5 * reco_img + 0.5 * img_input;
        } else {
            rectangle(img_input, pt1, pt2, Scalar(255.0, 255.0, 255.0), 1);
        }
        imshow("input", output);
    }
    if( pCap != nullptr ) delete pCap;  
  return 0;
}