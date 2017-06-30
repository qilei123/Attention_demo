#-------------------------------------------------
#
# Project created by QtCreator 2017-05-23T15:45:19
#
#-------------------------------------------------
QT       += core gui
#QT += multimedia multimediawidgets
CONFIG += c++11
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TARGET = Attention_Demo
TEMPLATE = app
# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
#below is opencv3.1.0
INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib -lopencv_attention -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_sfalign -lopencv_sfface -lopencv_sfident -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_core -lopencv_cudev -ldlib
#LIBS +=  -lFaceAnalyser -lLandmarkDetector
#INCLUDEPATH += /home/csuml/qilei_chen/ippicv_lnx/include
LIBS += /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
#LIBS += /usr/local/share/OpenCV/3rdparty/lib/liblibwebp.a
#below is opencv3.2.0 with seetaface and attention module
#CONFIG += c++11
#INCLUDEPATH += /home/olsen305/people_analytics/install_for_openface/include
#LIBS += -L/home/olsen305/people_analytics/install_for_openface/lib  -lopencv_calib3d -lopencv_attention -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_ml -lopencv_sfalign -lopencv_sfface -lopencv_sfident -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_objdetect -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_core -lopencv_cudev
#INCLUDEPATH += /home/olsen305/FACE/opencv_install/include
#LIBS += -L/home/olsen305/FACE/opencv_install/lib  -lopencv_calib3d -lopencv_attention -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_ml -lopencv_sfalign -lopencv_sfface -lopencv_sfident -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_objdetect -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_core -lopencv_cudev
#cudnn
INCLUDEPATH += /usr/local/cuda-8.0/include
LIBS += -L/usr/local/cuda-8.0/lib64 -lcudnn -lcudart
#tbb
INCLUDEPATH += /usr/include/tbb
LIBS += -L/usr/lib/x86_64-linux-gnu -ltbb -ltbbmalloc -ltbbmalloc_proxy
LIBS += -L/usr/lib/x86_64-linux-gnu -lglog -lgflags -lprotobuf -lprotobuf-lite -llmdb -lleveldb -lstdc++ -lcblas -latlas -lopenblas
LIBS += -ldl -fPIC -lpng12 -ljpeg -lz -lgobject-2.0 -lglib-2.0 -ldc1394 -ltiff -lcublas -lcurand -lprotobuf -pthread  -lpthread -lprotobuf-lite -pthread  -lpthread
#below is DLIB test in Qt
#INCLUDEPATH += /home/olsen305/DLIB/install_test/include
#LIBS += -L/home/olsen305/DLIB/install_test/lib -ldlib
LIBS += -lboost_filesystem -lboost_system -lboost_thread
LIBS += -lpng -ljasper -lgtk-x11-2.0 -lgdk-x11-2.0 -lpangocairo-1.0 -latk-1.0 -lcairo -lgdk_pixbuf-2.0 -lgio-2.0 -lpangoft2-1.0 -lpango-1.0
LIBS += -lfontconfig -lfreetype -lgobject-2.0 -lgmodule-2.0 -lgthread-2.0 -lglib-2.0
LIBS += -ldc1394 -lavcodec-ffmpeg -lavformat-ffmpeg -lavutil-ffmpeg -lswscale-ffmpeg -ldl -lm -lpthread -lrt -ltbb -latomic -lcudart -lnppc -lnppi -lnpps -lcufft -L/usr/local/cuda/lib64
#openface
#INCLUDEPATH += /home/olsen305/FACE/OpenFace/install_test/include
#LIBS += -L/home/olsen305/FACE/OpenFace/install_test/lib -lFaceAnalyser -lLandmarkDetector
#LIBS += /home/olsen305/FACE/OpenFace/install_test/lib/libFaceAnalyser.a
#LIBS += /home/olsen305/FACE/OpenFace/install_test/lib/libLandmarkDetector.a
#test rtpose
INCLUDEPATH +=/home/csuml/qilei_chen/caffe_rtpose/include
#INCLUDEPATH +=/home/olsen305/rtpose_ying_li/caffe_rtpose/src
LIBS += -L/home/csuml/qilei_chen/caffe_rtpose/.build_release/lib -lcaffe
SOURCES += main.cpp\
        mainwindow.cpp \
    analyticslabel.cpp \
    FaceAnalyser/src/Face_utils.cpp \
    FaceAnalyser/src/FaceAnalyser.cpp \
    FaceAnalyser/src/GazeEstimation.cpp \
    FaceAnalyser/src/SVM_dynamic_lin.cpp \
    FaceAnalyser/src/SVM_static_lin.cpp \
    FaceAnalyser/src/SVR_dynamic_lin_regressors.cpp \
    FaceAnalyser/src/SVR_static_lin_regressors.cpp \
    LandmarkDetector/src/CCNF_patch_expert.cpp \
    LandmarkDetector/src/LandmarkDetectionValidator.cpp \
    LandmarkDetector/src/LandmarkDetectorFunc.cpp \
    LandmarkDetector/src/LandmarkDetectorModel.cpp \
    LandmarkDetector/src/LandmarkDetectorParameters.cpp \
    LandmarkDetector/src/LandmarkDetectorUtils.cpp \
    LandmarkDetector/src/Patch_experts.cpp \
    LandmarkDetector/src/PAW.cpp \
    LandmarkDetector/src/PDM.cpp \
    LandmarkDetector/src/stdafx.cpp \
    LandmarkDetector/src/SVR_patch_expert.cpp \
    faceinitask.cpp \
    poseinitask.cpp \
    poseestimationtask.cpp \
    skeletonlabel.cpp \
    landmarktask.cpp \
    svm.cpp
HEADERS  += mainwindow.h \
    analyticslabel.h \
    FaceAnalyser/include/Face_utils.h \
    FaceAnalyser/include/FaceAnalyser.h \
    FaceAnalyser/include/GazeEstimation.h \
    FaceAnalyser/include/SVM_dynamic_lin.h \
    FaceAnalyser/include/SVM_static_lin.h \
    FaceAnalyser/include/SVR_dynamic_lin_regressors.h \
    FaceAnalyser/include/SVR_static_lin_regressors.h \
    LandmarkDetector/include/CCNF_patch_expert.h \
    LandmarkDetector/include/LandmarkCoreIncludes.h \
    LandmarkDetector/include/LandmarkDetectionValidator.h \
    LandmarkDetector/include/LandmarkDetectorFunc.h \
    LandmarkDetector/include/LandmarkDetectorModel.h \
    LandmarkDetector/include/LandmarkDetectorParameters.h \
    LandmarkDetector/include/LandmarkDetectorUtils.h \
    LandmarkDetector/include/Patch_experts.h \
    LandmarkDetector/include/PAW.h \
    LandmarkDetector/include/PDM.h \
    LandmarkDetector/include/stdafx.h \
    LandmarkDetector/include/SVR_patch_expert.h \
    faceinitask.h \
    poseinitask.h \
    poseestimationtask.h \
    skeletonlabel.h \
    landmarktask.h \
    svm.h
FORMS    += mainwindow.ui
