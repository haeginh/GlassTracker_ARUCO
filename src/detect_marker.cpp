/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,r
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd/depth.hpp>
#include <iostream>
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <AzureKinect.hh>
#include <run_options.hh>

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPLYReader.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTransform.h>
#include <vtkActor.h>
#include <opencv2/viz.hpp>
#include <opencv2/viz/widget_accessor.hpp>
#include <map>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

namespace {
const char* about = "Basic marker detection";
const char* keys  =
        "{d        | 0     | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16,"
        "DICT_APRILTAG_16h5=17, DICT_APRILTAG_25h9=18, DICT_APRILTAG_36h10=19, DICT_APRILTAG_36h11=20}"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{c        |       | Camera intrinsic parameters. Needed for camera pose }"
        "{l        | 0.1   | Marker side length (in meters). Needed for correct scale in camera pose }"
        "{dp       |       | File of marker detector parameters }"
        "{r        |       | show rejected candidates too }"
        "{refine   |       | Corner refinement: CORNER_REFINE_NONE=0, CORNER_REFINE_SUBPIX=1,"
        "CORNER_REFINE_CONTOUR=2, CORNER_REFINE_APRILTAG=3}";
}

class WPoly : public viz::Widget3D
{
public:
    WPoly(){}
    WPoly(const string & fileName);
    vtkAlgorithmOutput* GetPolyDataPort() {return reader->GetOutputPort();}
    void Initialize(const string & fileName);
    void Transform();
private:
    double* mat;
    vtkPLYReader* reader;
    vtkSmartPointer<vtkPolyDataMapper> mapper;
    vtkSmartPointer<vtkTransform> transform;
    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter;
};

/**
 * @function TriangleWidget::TriangleWidget
 * @brief Constructor
 */
WPoly::WPoly(const string & fileName)
{
    Initialize(fileName);
}
void WPoly::Initialize(const string &fileName){
    transform = vtkSmartPointer<vtkTransform>::New();
    transformFilter =
            vtkSmartPointer<vtkTransformPolyDataFilter>::New();

    vtkSmartPointer<vtkPolyData> polyData;
    reader = vtkPLYReader::New ();
    reader->SetFileName (fileName.c_str());
    reader->Update ();
    polyData = reader->GetOutput ();
    // Create mapper and actor
    mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Store this actor in the widget in order that visualizer can access it
    viz::WidgetAccessor::setProp(*this, actor);
    mat = new double[16];
}

void WPoly::Transform(){
    //mat[0] = {1,0,0,0,0,1,0,0,0,0,1,0};
    transform->SetMatrix(mat);
    //    double translate[3] = {0,0,label.GetDistance(class_id,template_id)};
    //    transform->Translate(translate);
    transformFilter->SetTransform(transform);
    transformFilter->SetInputConnection(reader->GetOutputPort());
    transformFilter->Update();
    mapper->SetInputData(transformFilter->GetOutput());
}

/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}


bool clicked(false);
Rect cropRect(0,0,0,0);
Point P1(0,0), P2(0,0);
bool calc=false;
void onMouseCropImage(int event, int x, int y, int f, void* param);
void PrintPLY(string file, Mat& color, Mat& depth, Mat& xy_table);
void PrintPLY(string file, Mat& cloud, Mat& color);
/**
 */
int DETECT_MARKER(int argc, char** argv) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 2) {
        parser.printMessage();
        return 0;
    }

    int dictionaryId = parser.get<int>("d");
    bool showRejected = parser.has("r");
    bool estimatePose = parser.has("c");
    float markerLength = 4;//parser.get<float>("l");

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }

    if (parser.has("refine")) {
        //override cornerRefinementMethod read from config file
        detectorParams->cornerRefinementMethod = parser.get<int>("refine");
    }
    std::cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): " << detectorParams->cornerRefinementMethod << std::endl;

    String video;
    if(parser.has("v")) {
        video = parser.get<String>("v");
    }

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    //    Ptr<aruco::Dictionary> dictionary =
    //            aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    Ptr<aruco::Dictionary> dictionary = aruco::generateCustomDictionary(6,4,2);
    //    Mat markerImg;
    //    for(int i=0;i<6;i++){
    //        aruco::drawMarker(dictionary, i, 200, markerImg, 1);
    //        imshow("marker", markerImg);
    //        waitKey(1000);
    //        getchar();
    //        //imwrite(out.substr(0,out.size()-4)+std::to_string(i)+out.substr(out.size()-4,4), markerImg);
    //    }

    Mat camMatrix, distCoeffs;
    if(estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }

    //    //read ply file
    //    viz::Viz3d myWindow("PLY viewer");
    //    WPoly poly;
    //    poly.Initialize("glass.ply");
    //    myWindow.showWidget("model PLY", poly);
    //    Vec3f cam_pos(0,0,-3000), cam_focal_point(0,0,1), cam_y_dir(0,-1.,0);
    //    Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_pointresize(imageCopy, imgResize, Size(1365,1024));, cam_y_dir);
    //    myWindow.setViewerPose(cam_pose);
    //    poly.Transform();
    //    for(int i=0;i<1000;i++){
    //        myWindow.spinOnce(1,true);
    //        cout<<"\rClick the widget to show the whole model.."<<i<<"/1000     "<<flush;
    //    }

    VideoCapture inputVideo;
    k4a::device device;
    k4a::transformation main_depth_to_main_color;
    Mat xy_table;
    int waitTime;
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    } else {
        device = k4a::device::open(0);

        // Start camera. Make sure depth camera is enabled.
        k4a_device_configuration_t deviceConfig = get_default_config();
        deviceConfig.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;

        // Get calibration information
        k4a::calibration main_calibration = device.get_calibration(deviceConfig.depth_mode,deviceConfig.color_resolution);
        main_depth_to_main_color = k4a::transformation(main_calibration);

        //xy table
        k4a_float2_t p;
        k4a_float3_t ray;
        int width_c = main_calibration.color_camera_calibration.resolution_width;
        int height_c = main_calibration.color_camera_calibration.resolution_height;
        xy_table=cv::Mat::zeros(height_c, width_c, CV_32FC2);
        float* xy_data = (float*)xy_table.data;

        //uchar
        for (int y = 0, idx = 0; y < height_c; y++)
        {
            p.xy.y = (float)y;
            for (int x = 0; x < width_c; x++, idx++)
            {
                p.xy.x = (float)x;

                if(main_calibration.convert_2d_to_3d(p,1.f,K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR, &ray))
                {
                    xy_data[idx*2] = ray.xyz.x;
                    xy_data[idx*2+1] = ray.xyz.y;
                }
            }
        }
        device.start_cameras(&deviceConfig);
        waitTime = 10;
    }

    double totalTime = 0;
    int totalIterations = 0;

    //viewer
    k4a::capture capture;
    map<int, vector<Point2f>> corner_cumul;
    vector<Vec3d> pose_cumul; //rvec XYZ, tvec
    int cumulCount(0);

    ofstream ofs("test.txt");
//    P1.x=2280; P1.y=1590;
//    P2.x=2664; P2.y=1857;
//    cropRect.x=P1.x;
//    cropRect.width=P2.x-P1.x;
//    cropRect.y=P1.y;
//    cropRect.height=P2.y-P1.y;
    vector<double> coeffX = {15,-1,-15,15,-1,-15};
    vector<double> coeffY = {-10,-10,-10,10,10,10};
    while(device.get_capture(&capture, std::chrono::milliseconds{ K4A_WAIT_INFINITE })) {
        //prepare image
        Mat image, imageCopy, cropImg, depthCopy, depthCrop;//cropImgCopy;
        k4a::image main_color_image = capture.get_color_image();
        k4a::image main_depth_image = capture.get_depth_image();
        k4a::image main_depth_in_main_color = create_depth_image_like(main_color_image);
        main_depth_to_main_color.depth_image_to_color_camera(main_depth_image, &main_depth_in_main_color);
        image = color_to_opencv(main_color_image);
        Mat depth = depth_to_opencv(main_depth_in_main_color);

        double tick = (double)getTickCount();

        setMouseCallback("out",onMouseCropImage,&imageCopy);

        //drawing rect
        if(clicked){
            cv::rectangle(image,P1,P2,CV_RGB(255,255,0),3);
            cv::rectangle(depth,P1,P2,CV_RGB(255,255,0),3);
            Mat imgResize, depthResize;
            resize(image, imgResize, Size(1365,1024));
            resize(depth, depthResize, Size(1365,1024));

            imshow("out", imgResize);
            // imshow("depth", depthResize*10);
            waitKey(waitTime);
            continue;
        }

        // crop the image
        if(cropRect.width>0){
            cropImg = image(cropRect).clone();
            depthCrop = depth(cropRect).clone();
        }else{
            image.copyTo(cropImg);
            depth.copyTo(depthCrop);
        }

        // detect markers
        vector< int > ids;
        vector< vector< Point2f > > cornersCrop, cornersWhole, rejected;
        aruco::detectMarkers(cropImg, dictionary, cornersCrop, ids, detectorParams, rejected);

        // compare with the previous result
        if(ids.size()>0){
            bool isNewPose(false);
            for(int i=0;i<ids.size();i++) {
                vector< Point2f > points;
                for(auto p:cornersCrop[i]) points.push_back(Point2f(p.x+P1.x,p.y+P1.y));
                cornersWhole.push_back(points);
                if(isNewPose) continue;
                if(corner_cumul.find(ids[i])!=corner_cumul.end()){
                    Point2f oldCen(0,0), newCen(0,0);
                    for(int n=0;n<4;n++) {
                        newCen  += points[n];
                        oldCen  += corner_cumul[ids[i]][n];
                    }
                    Point2f vec = oldCen-newCen;
                    if(vec.dot(vec)>40) {
                        corner_cumul.clear();
                        //pose_cumul.clear();
                        isNewPose = true;
                        cumulCount = 0;
                        //                       cout<<endl<<"new Pose!------------------------"<<endl;
                    }else cumulCount++;
                }
            }
            for(int i=0;i<ids.size();i++) corner_cumul[ids[i]] = cornersWhole[i];
        }else {
            corner_cumul.clear();
            cumulCount = 0;
        }

        //draw result
        image.copyTo(imageCopy);
        depth.copyTo(depthCopy);
        if(corner_cumul.size()>0){
            vector< vector< Point2f > > cornersTemp;
            vector< int > idsTemp;
            for(auto iter:corner_cumul){
                cornersTemp.push_back(iter.second);
                idsTemp.push_back(iter.first);
            }
            aruco::drawDetectedMarkers(imageCopy, cornersTemp, idsTemp);
            aruco::drawDetectedMarkers(depthCopy, cornersTemp, idsTemp);
        }


        //draw mask
/*        Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);
        uint16_t* depth_data = (uint16_t*)depth.data;
        float* xy_data = (float*) xy_table.data;
        vector<Eigen::Vector3f> centroidVec;
        for(int n=0;n<ids.size();n++){
            vector< Point2f > p = cornersWhole[n];
            //make small rect
            Point p0(p[0]),p1(p[1]),p2(p[2]),p3(p[3]);
            Point center = (p0+p1+p2+p3)*0.25;
            vector<Point> pp = {(p0+center)*0.5,(p1+center)*0.5,(p2+center)*0.5,(p3+center)*0.5};
            const Point* elementPoints[1] = { &pp[0] };
            int numPoints = (int)pp.size();
            Mat mask1 = cv::Mat::zeros(image.rows, image.cols, CV_8U);
            fillPoly(mask1, elementPoints, &numPoints, 1, Scalar(255)); mask += mask1;
            Mat masked1 = cv::Mat::zeros(image.rows, image.cols, CV_16U);
            bitwise_and(depth,depth, masked1, mask1);

            //set the bounding box
            int maxX(center.x), minX(center.x), maxY(center.y), minY(center.y);
            for(int i=0;i<4;i++){
                if(p[i].x>maxX) maxX = p[i].x;
                else if(p[i].x<minX) minX = p[i].x;
                if(p[i].y>maxY) maxY = p[i].y;
                else if(p[i].y<minY) minY = p[i].y;
            }
            //find center
            uchar* mask_data = (uchar*) mask1.data;
            int num = countNonZero(masked1);
            if(num==0) {centroidVec.push_back(Eigen::Vector3f(0,0,0)); continue;}
            Eigen::MatrixXf points(3,num);
            for(int j=minY, m=0;j<maxY;j++){
                for(int i=minX;i<maxX;i++){
                    int idx = j*mask1.cols + i;
                    if(mask_data[idx]==0) continue;
                    float z=(float)depth_data[idx]*0.1;
                    if(z==0) continue;
                    points(0,m)=z*xy_data[idx*2];
                    points(1,m)=z*xy_data[idx*2+1];
                    points(2,m)=z;
                    m++;
                }
            }
            Eigen::Vector3f centroid(points.row(0).mean(), points.row(1).mean(), points.row(2).mean());
            //            cout<<ids[n]<<"centroid: "<<centroid.transpose()<<endl;
            centroidVec.push_back(centroid);

            //            points.row(0).array() -= centroid(0); points.row(1).array() -= centroid(1); points.row(2).array() -= centroid(2);
            //            auto svd = points.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
            //            Eigen::Vector3f plane_normal = svd.matrixU().rightCols<1>();
            //            cout<<plane_normal<<endl;
        }*/

        // 2D pose estimation
        Affine3d rot;
        if(estimatePose && ids.size() > 0){
            vector< Vec3d > rvecs, tvecs;
            aruco::estimatePoseSingleMarkers(cornersWhole, markerLength, camMatrix, distCoeffs, rvecs,
                                             tvecs);

            Vec3d tvec(0,0,0),axisX(0,0,0),axisY(0,0,0), axisZ(0,0,0);

            // six variations (x -> z -> y)
            for(int i=0;i<ids.size();i++){
                Affine3d rot(rvecs[i]);
                axisX += rot*Vec3d(1,0,0);
                axisY += rot*Vec3d(0,1,0);
//                axisZ += rot*Vec3d(0,0,1);
            }

//            ofs<<endl;
            if(cumulCount>1){
                axisX += pose_cumul[0]*(cumulCount-1);
                axisY += pose_cumul[1]*(cumulCount-1);
            }
            axisX = normalize(axisX); axisY = normalize(axisY);
            axisZ = axisX.cross(axisY);
            axisY = axisZ.cross(axisX);
            double rotData[9] = {axisX(0),axisY(0),axisZ(0),axisX(1),axisY(1),axisZ(1),axisX(2),axisY(2),axisZ(2)};
            Mat rotMat = cv::Mat(3,3,CV_64F,rotData);
             rot.rotation(rotMat);

            for(int i=0;i<ids.size();i++){
                int id = ids[i];
                Vec3d xTrans = coeffX[id]*axisX;
                Vec3d yTrans = coeffY[id]*axisY;
  //              cout<<id<<tvecs[i]<<endl;
                tvec += (tvecs[i]+xTrans+yTrans);
            }
            tvec *= 1.f/ids.size();
            if(cumulCount>1){
 //               cout<<tvec<<endl;
                tvec += pose_cumul[3]*(cumulCount-1);
                tvec /= (double)cumulCount;
            }
           // cout<<tvec<<endl;
            aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rot.rvec(), tvec,
                            markerLength * 3.f);


            pose_cumul = {axisX, axisY, axisZ, tvec};
             //            cout<<"original"<<endl;
//            cout<<axisX<<endl<<axisY<<endl<<axisZ;


//            Eigen::Vector3f centroid(points.row(0).mean(), points.row(1).mean(), points.row(2).mean());
//            points.row(0).array() -= centroid(0); points.row(1).array() -= centroid(1); points.row(2).array() -= centroid(2);
//            auto svd = points.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
//            Eigen::Vector3f plane_normal = svd.matrixU().rightCols<1>();

//            for(int i=0;i<ids.size();i++)
//            aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
//                            markerLength * 0.5f);

//            map<int, pair<Vec3d, Vec3d >> markerPose;
//            for(int i=0;i<ids.size();i++){
//                markerPose[ids[i]] = make_pair(rvecs[i],tvecs[i]);
//            }
//            for(int i=0;i<6;i++){
//                auto iter = markerPose.find(i);
//                if(iter==markerPose.end())
//                    ofs<<"\t\t\t\t\t\t";rvec
//                else
//                    ofs<<iter->second.first<<iter->second.second;
//            }ofs<<endl;
        }
        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }


        //    Mat masked=cv::Mat::zeros(image.rows, image.cols, CV_16U);
        //    bitwise_and(depth,depth, masked, mask);
        //    cv::rectangle(imageCopy,P1,P2,CV_RGB(255,255,0),3);
        //   cv::rectangle(depthCopy,P1,P2,Scalar(60000),3);
        //   // resize(depthCopy, depthResize, Size(1365,1024));

        //    Mat maskResize;
        //    if(cropRect.width==0) resize(masked, maskResize, Size(1365,1024));
        //    else if(cropRect.height>1024) resize(masked, maskResize, Size((double)cropRect.width/(double)cropRect.height*1024,1024));
        //    else if(cropRect.width>1365) resize(masked, maskResize, Size(1365,(double)cropRect.height/(double)cropRect.width*1365));
        //    else maskResize = masked;
        //    maskResize*=10;
        //    putText(maskResize, to_string(masked.cols)+"*"+to_string(masked.rows),
        //            Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(60000), 1);

        Mat imgResize;
        cv::rectangle(imageCopy,P1,P2,CV_RGB(255,255,0),3);
        resize(imageCopy, imgResize, Size(1365,1024));
        putText(imgResize, to_string(cropRect.width)+"*"+to_string(cropRect.height)+" ("+to_string(P1.x)+", "+to_string(P1.y)+" / "+to_string(P2.x)+", "+to_string(P2.y)+")",
                Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(60000), 1);
        stringstream ss_tvec; ss_tvec<<"trans: "<<pose_cumul[3];
        stringstream ss_rvec; ss_rvec<<"rvec : "<<rot.rvec();
        stringstream ss_count; ss_count<<"accumulated data : "<<cumulCount;
        putText(imgResize, ss_tvec.str(),
                Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(60000), 1);
        putText(imgResize, ss_rvec.str(),
                Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(60000), 1);
        putText(imgResize, ss_count.str(),
                Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(60000), 1);
        imshow("out", imgResize);
        if(cropRect.width>0){
            imshow("crop_color", imageCopy(cropRect));
            imshow("crop_depth", depthCopy(cropRect)*10);
        }

        // imshow("depth", depthResize*10);
        char key = (char)waitKey(waitTime);
        if (key == 'q')
            break;

        switch (key)
        {
        case 's':
            calc=true;
            break;
        default:
            ;
        }
    }
    ofs.close();
    return 0;
}

void onMouseCropImage(int event, int x, int y, int f, void *param){
    switch (event) {
    case EVENT_LBUTTONDOWN:
        clicked = true;
        P1.x = x*3;
        P1.y = y*3;
        P2.x = x*3;
        P2.y = y*3;
        break;
    case EVENT_LBUTTONUP:
        P2.x=x*3;
        P2.y=y*3;
        clicked = false;
        break;
    case EVENT_MOUSEMOVE:
        if(clicked){
            P2.x=x*3;
            P2.y=y*3;
        }
        break;
    case EVENT_RBUTTONUP:
        clicked = false;
        P1.x = 0;
        P1.y = 0;
        P2.x = 0;
        P2.y = 0;
        break;
    default:
        break;
    }

    if(clicked){
        if(P1.x>P2.x){
            cropRect.x=P2.x;
            cropRect.width=P1.x-P2.x;
        }
        else{
            cropRect.x=P1.x;
            cropRect.width=P2.x-P1.x;
        }

        if(P1.y>P2.y){
            cropRect.y=P2.y;
            cropRect.height=P1.y=P2.y;
        }
        else{
            cropRect.y=P1.y;
            cropRect.height=P2.y-P1.y;
        }
    }
}

void PrintPLY(string file, Mat& color, Mat& depth, Mat& xy_table){
    vector<vector<double>> xyz;
    vector<vector<int>> rgb;
    uint16_t *depth_data = (uint16_t *)depth.data;
    uint8_t  *color_data = (uint8_t  *)color.data;
    float*    xy_data    = (float*)xy_table.data;
    for(int y=0, idx = 0;y<depth.rows; y++){
        for(int x=0;x<depth.cols; x++,idx++){
            if(depth_data[idx]==0) continue;
            float z=depth_data[idx]*0.1;
            xyz.push_back({xy_data[idx*2]*z,xy_data[idx*2+1]*z,z});
            rgb.push_back({color_data[idx*3],color_data[idx*3+1],color_data[idx*3+2]});
        }
    }

    ofstream ofs(file);
    ofs<<"ply"<<endl;
    ofs<<"format ascii 1.0"<<endl;
    ofs<<"comment exported in ArUco_detect"<<endl;
    ofs<<"element vertex "<<xyz.size()<<endl;
    ofs<<"property float x"<<endl;
    ofs<<"property float y"<<endl;
    ofs<<"property float z"<<endl;
    ofs<<"property uchar red"<<endl;
    ofs<<"property uchar green"<<endl;
    ofs<<"property uchar blue"<<endl;
    ofs<<"end_header"<<endl;
    for(int i=0;i<xyz.size();i++)
        ofs<<xyz[i][0]<<" "<<xyz[i][1]<<" "<<xyz[i][2]<<" "<<rgb[i][0]<<" "<<rgb[i][1]<<" "<<rgb[i][2]<<endl;
    ofs.close();
}

void PrintPLY(string file, Mat& depth, Mat& xy_table){
    vector<vector<double>> xyz;
    uint16_t *depth_data = (uint16_t *)depth.data;
    float*    xy_data    = (float*)xy_table.data;
    for(int y=0, idx = 0;y<depth.rows; y++){
        for(int x=0;x<depth.cols; x++,idx++){
            if(depth_data[idx]==0) continue;
            float z=depth_data[idx]*0.1;
            xyz.push_back({xy_data[idx*2]*z,xy_data[idx*2+1]*z,z});
        }
    }

    ofstream ofs(file);
    ofs<<"ply"<<endl;
    ofs<<"format ascii 1.0"<<endl;
    ofs<<"comment exported in ArUco_detect"<<endl;
    ofs<<"element vertex "<<xyz.size()<<endl;
    ofs<<"property float x"<<endl;
    ofs<<"property float y"<<endl;
    ofs<<"property float z"<<endl;
    ofs<<"end_header"<<endl;
    for(int i=0;i<xyz.size();i++)
        ofs<<xyz[i][0]<<" "<<xyz[i][1]<<" "<<xyz[i][2]<<endl;
    ofs.close();
}

//void PrintPLY(string file, Mat& cloud, Mat& color){
//    ofstream ofs(file);
//    ofs<<"ply"<<endl;
//    ofs<<"format ascii 1.0"<<endl;
//    ofs<<"comment exported in ArUco_detect"<<endl;
//    ofs<<"element vertex "<<cloud.rows<<endl;
//    ofs<<"property float x"<<endl;
//    ofs<<"property float y"<<endl;
//    ofs<<"property float z"<<endl;
//    ofs<<"property uchar red"<<endl;
//    ofs<<"property uchar green"<<endl;
//    ofs<<"property uchar blue"<<endl;
//    ofs<<"end_header"<<endl;
//    float* cloud_data = (float*) cloud.data;
//    uchar* color_data = (uchar*) color.data;
//    for(int i=0;i<cloud.rows;i++)
//        ofs<<cloud_data[i*3]<<" "<<cloud_data[i*3+1]<<" "<<cloud_data[i*3+2]<<" "<<(int)color_data[i*3]<<" "<<(int)color_data[i*3+1]<<" "<<(int)color_data[i*3+2]<<endl;
//    ofs.close();
//}
