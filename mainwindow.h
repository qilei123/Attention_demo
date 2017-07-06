#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QKeyEvent>
#include <QTimer>
#include <QString>
#include <QFileDialog>
#include <QDir>
#include <QMessageBox>
#include <QImage>
#include <QDebug>
#include <QResizeEvent>
#include <QThreadPool>
#include <QPainter>
#include <QPen>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/sfface.hpp>
#include <opencv2/sfalign.hpp>
#include <opencv2/sfident.hpp>

#include <tbb/atomic.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <queue>
#include <cmath>

#include <LandmarkDetector/include/LandmarkCoreIncludes.h>
//--------------------ying_li_head-------------------------
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <utility> //std::pair

#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <stdio.h>  // snprintf
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

#include <boost/thread/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
// #include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/cpm/frame.h>
#include <caffe/cpm/layers/imresize_layer.hpp>
#include <caffe/cpm/layers/nms_layer.hpp>
#include <caffe/net.hpp>
#include <caffe/util/blocking_queue.hpp>

#include <rtpose/modelDescriptor.h>
#include <rtpose/modelDescriptorFactory.h>
#include <rtpose/renderFunctions.h>
#include <rtpose/rtPose.hpp>
#include <json/json.h>
//--------------------ying_li_tail-------------------------

#include "analyticslabel.h"
#include "svm.h"

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

#define NO_SRC 0
#define CAMERA 1
#define VIDEO 2

//pose estimation life state
#define POSEBORN -1
#define POSEREADY 0
#define POSERUNNING 1
#define POSESTOPED 2

#define PI 3.1415926    //define π

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

protected:
     void keyPressEvent(QKeyEvent *);
     //void resizeEvent(QResizeEvent *event);
private:
    Ui::MainWindow *ui;
    AnalyticsLabel *analytics_label;
    QTimer *main_timer;
    cv::VideoCapture cap;
    //camera resolution
    int h = 480;
    int w = 640;
    /* resolution list
     * 1920 1080
     * 1504 832
     * 1280 720
     * 1024 768
     * 960 720
     * 640 480
     * 352 288
     * 320 240
     * 176 144
     * 160 120
     */
    cv::Mat main_frame;
    cv::Mat original_frame;

    bool runPoseState = false;
    int poseEstState = POSEBORN;
    std::queue<cv::Mat> poseImgQueue;
    int queueMaxSize = 5;

    int src = NO_SRC;

    std::string filename;
    QSize qdisplay_size;
    QSize pre_label_img_size;
    cv::sfface::sffaceRecognizer *seetaRecognizer;

    //std::string SDATA_DIR = "/home/olsen305/penfei/people_analytics-face_dev/seeta_testcases/cv_facedetection/data/";
    std::string SMODEL_DIR = "/home/olsen305/penfei/models/model_detection/";

    float fx = 600, fy = 600, cx = 0, cy = 0;
    LandmarkDetector::FaceModelParameters *det_params;
    vector<LandmarkDetector::FaceModelParameters> det_parameters;
    vector<LandmarkDetector::CLNF> clnf_models;
    vector<bool> active_models;
    int num_faces_max = 5;
    LandmarkDetector::CLNF *clnf_model;
    bool cx_undefined = false;

    int frame_count = 0;
    vector<cv::Rect_<double> > face_detections;
    // For measuring the timings
    int64 t1,t0;
    double fps = 10;

    bool all_models_active = true;

    QLabel *statusBarLabel;

    int iniStep = 0;
    RTPose *realPose;

    QPainter *skeleton_painter;
    QPen *skeleton_pen;

    int joints_num = 18;

    cv::Point mouse_position;

    float mn_distance_thredshold = 50;

    bool skeleton_show_one = false;
    bool face_show_one = false;

    bool face_showable = true;
    bool emotion_showable = true;//unimplementation
    bool skeleton_showable = true;
    bool landmark_showable = true;

    int thread_strategy = 0;

    int processStop =0;
private slots:
    void slot_open_video();      // open video
    void slot_open_video_file(); //open video file
    void slot_read_frame();       // read frame
    void slot_close_video();     // close video
    void slot_exit();
    void slot_faces_check();
    void slot_emotion_check();
    void slot_skeleton_check();
    void slot_landmark_check();

public slots:
    void slot_press_position(int,int);

signals:
    void sig_paintSkeleton();
    void sig_paintLandmark();
private slots:
    void slot_paintSkeleton();
    void slot_paintLandmark();
private:
    QSize imageDisplaySize(cv::Size &src_size,QSize &label_img_size);
    std::vector<seeta::FaceInfo> faceDetection(cv::Mat src);

    void NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double> >& face_detections);
    int landmarkDetect(cv::Mat &captured_image,vector<cv::Rect_<double> >& face_detections);
    int landmarkDetect1(cv::Mat &captured_image, vector<cv::Rect_<double> > &face_detections);
    void transSeetaface2Rect(std::vector<seeta::FaceInfo> &seetaFaces, std::vector<cv::Rect_<double> >& face_detections);
    void iniReady(QString tips);

    void drawJoints(cv::Mat src,int positions[][2]);
    void drawBones(cv::Mat src, int positions[][2]);
    void drawLine(cv::Mat src,cv::Point start_point,cv::Point end_point);
    void to_point(cv::Point &start_point,cv::Point &end_point,int positions[][2],int start_index,int end_index);
    float mouse_nose_distance(int positions[][2]);

    int angleAttention(int base_vector[2],int reality_vector[2]);
    double calAngle(int a[2],int b[2]);
    int getAngleAttention(int base_vector[2],int positions[][2],int index1,int index2);

    int headAttention = 0;
    int head_base[2]={0,-1};

    int shoulderAttention = 0;
    int shoulder_base[2] = {1,0};

    int eyesAttention = 0;
    int mouthAttention = 0;
    int faceAttention = 0;
    int getFaceAttention(cv::Vec6d &pose_estimate);
    int getEyesAttention(vector<cv::Point2d> faceLandMarks);
    int getMouthAttention(vector<cv::Point2d> faceLandMarks);

    //double two_points_distance(double marks[][2],int index1,int index2);

    cv::Mat landmark_frame;

    //emotion
    svm_model *svmModel;
    vector<svm_node*> features;
    vector<string> emotion_predictions;

    double getDist(cv::Point2f p1, cv::Point2f p2);
    void landmarks2features(LandmarkDetector::CLNF *clnf_model, svm_node* node);
    string result2labels(int class_nr_int);

    int emotionStyle = 0;
    string emotion_des;

    double imagescale = 1;

    string outputVideoName = "output.avi";
    int frameRate = 25;
    int codec = 0;
    cv::VideoWriter writer;
    int camera_id = 0;
private:
    QImage scaledImage_skeleton;
    QImage scaledImage_landmark;
public:
     bool landmark_running = false;
     void landmarkDetect();
     bool set_landmark_running(bool st);

     void landmarkInit();
     void notify_face_ini_ok();
     void poseInit();
     void notify_pose_ini_ok();

     //void setRunPoseState(bool state);
     void runPoseEst();
    //void cursorPosCallBack(int x, int y);
};

#endif // MAINWINDOW_H
