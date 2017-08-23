# Attention_demo
System requirement:
Ubuntu 16.04.x (better a clean ubuntu machine)
Nvidia gtx 10xx
cudnn 5.0 (not 5.1 or 6)
use a special config opencv(opencv-3.1.0_cql.tar.gz)
g++ 5.0 (the system should not install more than one edition g++)
use a special config dlib(dlib-19.4_cql.tar.gz)

step1:
setup cuda, reference:
install cuda: http://www.linuxdiyf.com/linux/28353.html
add 
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
in the file ~/.bashrc and then source ~/.bashrc

step2:
build and install opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_IPP=ON -D WITH_WEBP=ON -D WITH_TBB=ON -D BUILD_SHARED_LIBS=ON ..
make -j8
sudo make install -j8

step3:
build and install dlib:
mkdir build
cd build
 cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_SHARED_LIBS=ON ..
make -j8
sudo make install -j8

step4:
OpenPose Setup:
install caffe first
and modify the Makefile.config in openpose:
CAFFE_DIR := 3rdparty/caffe/distribute  ===>   CAFFE_DIR:= yourpathtocaffefolder
then use command:
make -j8
to compile the openpose project.
Go to folder models in openpose, run getModels.sh to download the model.
Then you could run ./build/example/openpose/openpose.bin
Before you build openpose, you can change some parameter in the file examples/openpose/openpose.cpp to adjust the result of openpose.

Step5:
build rtpose:
go in the folder caffe_rtpose and just build is ok:
make -j8
step6:
build and run Attention_demo:
Go in the folder Attention_demo, change some paths in the file called Attention_Demo.pro:
INCLUDEPATH +=/yourpath/caffe_rtpose/include
LIBS += -L/yourpath/caffe_rtpose/.build_release/lib -lcaffe
Then:
mkdir build
cd build
qmake /home/csuml/qilei_chen/Attention_Demo/Attention_Demo.pro -r -spec linux-g++ CONFIG+=debug CONFIG+=qml_debug
make -j8
then unzip models.tar.gz  and copy all the files into the folder build
use ./Attention_demo to run it.
Also there are some parameter in the file of mainwindow.h that you can modify:
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
which can control the input camera resolution.
    int camera_id = 0;// the id that you want to use, 0 is the default camera
    int num_faces_max = 20;// it means that it can detect as many as 20 face’s landmark.
Usefull tips:
add some common lib into the file ~/.bashrc will help to run the project.
export LD_LIBRARY_PATH=/home/csuml/Downloads/caffe-master/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/home/csuml/qilei_chen/caffe_rtpose/.build_release/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
