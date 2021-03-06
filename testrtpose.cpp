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

#include "caffe/cpm/frame.h"
#include "caffe/cpm/layers/imresize_layer.hpp"
#include "caffe/cpm/layers/nms_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/blocking_queue.hpp"
// #include "caffe/util/render_functions.hpp"
// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/db.hpp"
// #include "caffe/util/io.hpp"
// #include "caffe/util/benchmark.hpp"

#include "rtpose/modelDescriptor.h"
#include "rtpose/modelDescriptorFactory.h"
#include "rtpose/renderFunctions.h"
#include "rtpose/rtPose.hpp"

void *testrtpose(void* args) {

    RTPose rp("/home/olsen305/rtpose_ying_li/caffe_rtpose/model/coco/pose_iter_440000.caffemodel", "/home/olsen305/rtpose_ying_li/caffe_rtpose/model/coco/pose_deploy_linevec.prototxt", 0);

    //RTPose rp("/home/olsen305/rtpose_ying_li/caffe_rtpose/model/mpi/pose_iter_160000.caffemodel", "/home/olsen305/rtpose_ying_li/caffe_rtpose/model/mpi/pose_deploy_linevec.prototxt", 0);

        cv::VideoCapture cap(0); // open the default camera
        if(!cap.isOpened())  // check if we succeeded
            ;
        //return;
        int count = 0;
        cv::Mat edges;
        //cv::namedWindow("test",1);
        for(;;)
        {
            cv::Mat frame;
            cap >> frame; // get a new frame from camera
            std::string res_json = rp.getPoseEstimation(frame);
            std::cout<<res_json<<std::endl;
            //cv::imshow("test", frame);
            std::cout<<count<<std::endl;
            count++;
            //if(cv::waitKey(30) >= 0) break;
        }
        cap.release();
        cv::destroyAllWindows();
//    cv::Mat img_mat = cv::imread("/home/olsen305/rtpose_ying_li/0630450043.jpg", CV_LOAD_IMAGE_COLOR);
////    cv::Mat img_small ;
////    cv::resize( img_mat, img_small, cv::Size(img_mat.cols / 5, img_mat.rows / 5) );
////    cv::imshow("test",img_small);
////    cv::waitKey(1);

//    std::string res_json = rp.getPoseEstimation(img_mat);

//    std::cout << "Output1 json result.\n";

//    std::cout << res_json;

//    std::cout << "\n";

//    cv::Mat img_mat2 = cv::imread("/home/olsen305/rtpose_ying_li/0630450043.jpg", CV_LOAD_IMAGE_COLOR);

//    std::string res_json2 = rp.getPoseEstimation(img_mat2);

//    std::cout << "Output2 json result.\n";

//    std::cout << res_json2;

//    std::cout << "\n";

}
