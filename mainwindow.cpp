#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "faceinitask.h"
#include "poseinitask.h"
#include "poseestimationtask.h"
#include "landmarktask.h"

class FaceIniTask;
class PoseIniTask;
class PoseEstimationTask;

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    analytics_label = new AnalyticsLabel();
    ui->gridLayout_6->removeWidget(ui->label_image);
    ui->gridLayout_6->addWidget(analytics_label);
    main_timer = new QTimer(this);
    mouse_position.x = -1000;
    mouse_position.y = -1000;
    connect(ui->pushButton_camera,SIGNAL(clicked()),this,SLOT(slot_open_video()));
    connect(ui->pushButton_video,SIGNAL(clicked()),this,SLOT(slot_open_video_file()));
    connect(main_timer,SIGNAL(timeout()),this,SLOT(slot_read_frame()));
    connect(ui->pushButton_stop,SIGNAL(clicked()),this,SLOT(slot_close_video()));
    connect(ui->pushButton_exit,SIGNAL(clicked()),this,SLOT(slot_exit()));

    connect(ui->check_face,SIGNAL(clicked()),this,SLOT(slot_faces_check()));
    connect(ui->check_emotion,SIGNAL(clicked()),this,SLOT(slot_emotion_check()));
    connect(ui->check_skeleton,SIGNAL(clicked()),this,SLOT(slot_skeleton_check()));
    connect(ui->check_landmark,SIGNAL(clicked()),this,SLOT(slot_landmark_check()));

    connect(this,SIGNAL(sig_paintSkeleton()),this,SLOT(slot_paintSkeleton()));
    connect(this,SIGNAL(sig_paintLandmark()),this,SLOT(slot_paintLandmark()));

    //seetaRecognizer = new cv::sfface::sffaceRecognizer((SMODEL_DIR + "seeta_fd_frontal_v1.0.bin").c_str());
    seetaRecognizer = new cv::sfface::sffaceRecognizer("seeta_fd_frontal_v1.0.bin");
    seetaRecognizer->SetMinFaceSize(40);
    seetaRecognizer->SetScoreThresh(2.f);
    seetaRecognizer->SetImagePyramidScaleFactor(0.8f);
    seetaRecognizer->SetWindowStep(4, 4);

    pre_label_img_size = analytics_label->size();

    //ui->skeleton_graphicsView->sceneRect().setSize(ui->widget_skeleton->size().width()-2,ui->widget_skeleton->size().height()-2);
    //ui->skeleton_graphicsView->scene()->

    statusBarLabel = new QLabel("Initializing...");
    ui->statusBar->addWidget(statusBarLabel);
    //QAction *label_ima
    //ui->label_image->addAction();
    //this->landmarkInit();
    ui->pushButton_camera->setDisabled(true);
    ui->pushButton_video->setDisabled(true);
    FaceIniTask *face_ini_task = new FaceIniTask(this);
    PoseIniTask *pose_ini_task = new PoseIniTask(this);
    face_ini_task->start();
    pose_ini_task->start();

}
MainWindow::~MainWindow()
{
    delete det_params;
    delete clnf_model;
    delete svmModel;
    delete realPose;
    delete main_timer;
    delete seetaRecognizer;
    delete analytics_label;
    delete ui;
}
void MainWindow::keyPressEvent(QKeyEvent *event)
{

    if(event->key()==Qt::Key_Escape)
    {
        this->slot_exit();
    }
    else if(event->key()==Qt::Key_F)
    {
        //full screen
        QMainWindow::showFullScreen();
        //pre_label_img_size = ui->label_image->size();
    }
    else if(event->key()==Qt::Key_Q)
    {
        QMainWindow::showMaximized();
//        ui->label_image->clear();
//        qdisplay_size = pre_label_img_size;
//        ui->label_image->resize(pre_label_img_size);
    }
}
//void MainWindow::resizeEvent(QResizeEvent *event)
//{
//    QMainWindow::resizeEvent(event);
//    qdisplay_size = ui->label_image->size();
//}
void MainWindow::slot_open_video()
{


    std::cout<<"open video"<<std::endl;
    if(this->src == NO_SRC)
    {
        this->src = CAMERA;
    }

    if(this->src == CAMERA)
    {
        cap.open(camera_id);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, h);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, w);

    }
    else if(this->src == VIDEO)
    {
        //cap.open("/home/olsen305/Kindergarten/11.mp4");
        cap.open(this->filename);
        cv::Mat teframe;
        cap>>teframe;
        w = teframe.cols/imagescale;
        h = (teframe.rows)/imagescale;
        std::cout<<w<<"-------"<<h<<std::endl;

    }
    cv::Size frameSize(w,h);
    if(!writer.open(outputVideoName, CV_FOURCC('D','I','V','X') , frameRate, frameSize, true))
    {
        std::cout<<"open writer error..."<<std::endl;
        this->slot_close_video();
    }

    if(cap.isOpened())
    {
        main_timer->start(10);
    }
    else
    {
        this->slot_close_video();
    }
    //qdisplay_size = ui->label_image->size();

    disconnect(ui->pushButton_camera,SIGNAL(clicked()),this,SLOT(slot_open_video()));
    disconnect(ui->pushButton_video,SIGNAL(clicked()),this,SLOT(slot_open_video_file()));

    connect(analytics_label,SIGNAL(signal_press_position(int,int)),this,SLOT(slot_press_position(int,int)));
    connect(analytics_label,SIGNAL(signal_reset_position(int,int)),this,SLOT(slot_press_position(int,int)));

    if(this->poseEstState==POSESTOPED||this->poseEstState==POSEBORN)
        this->poseEstState = POSEREADY;

}
void MainWindow::slot_open_video_file()
{

    QString qfilename = QFileDialog::getOpenFileName(
       this,
       "Open Document",
       QDir::currentPath(),
       "Document files (*.mp4 *.avi *.mov);;All files(*.*)");

    if (!qfilename.isNull())
    {    //select file
        // deal with file

        //QMessageBox::information(this, "Document", "get document", QMessageBox::Ok);

        std::cout<<qfilename.toStdString()<<std::endl;

        filename = qfilename.toUtf8().constData();

        this->src = VIDEO;
    }
    else // cancel choose
    {
        //QMessageBox::information(this, "Document", "no document", QMessageBox::Ok);
        return ;
    }

    slot_open_video();

    processStop = 0;

}
void MainWindow::slot_read_frame()
{

    //cap>>main_frame;
    cv::Mat orig;
    cap>>orig;

    //if(main_frame.data==NULL)
    if(orig.data==NULL)
    {
        slot_close_video();
        return;
    }
    cv::Size newsize(w,h);
    cv::resize(orig,main_frame,newsize);
    //if(this->poseEstState==POSERUNNING)
        if(this->poseImgQueue.size()<queueMaxSize){
            original_frame = main_frame.clone();
            poseImgQueue.push(original_frame);
        }
        else if(poseImgQueue.size() >= queueMaxSize)
        {
            poseImgQueue.pop();
        }
    //this->poseEstState = POSEREADY;
    if(this->poseEstState == POSEREADY)
    {
        this->poseEstState = POSERUNNING;
        PoseEstimationTask *pet = new PoseEstimationTask(this);
        pet->start();
    }

//    if(!runPoseState)
//    {
//        original_frame = main_frame.clone();
//        runPoseState = true;
//        PoseEstimationTask *pet = new PoseEstimationTask(this);
//        pet->start();
//    }

    //this->faceDetection(main_frame);
    //vector<cv::Rect_<double> > face_detections;

    std::vector<seeta::FaceInfo> faces = this->faceDetection(main_frame);


    if(!landmark_running){
        cv::Mat temp_landmark_frame;

//        if(!landmark_frame.empty())
//        {
//            temp_landmark_frame = landmark_frame.clone();
//        }

        landmark_frame = main_frame.clone();

//        if(!temp_landmark_frame.empty())
//        {
//            main_frame = temp_landmark_frame.clone();
//        }
        this->face_detections.clear();
        this->transSeetaface2Rect(faces,this->face_detections);

        //std::cout<<"----------------------faces.size()::::"<<faces.size()<<std::endl;
        //if(faces.size()>0){

            if(landmark_showable){
                landmark_running = true;
                LandmarkTask *landmarkTask = new LandmarkTask(this);
                landmarkTask->start();
            }
        //}
    }
    else
    {
        //draw land mark in the image
    }
    cv::Mat temp_main_frame;
    writer.write(main_frame);
    cv::cvtColor(main_frame, temp_main_frame,CV_BGR2RGB);
    QImage image = QImage((uchar*) temp_main_frame.data, temp_main_frame.cols, temp_main_frame.rows, temp_main_frame.step, QImage::Format_RGB888);
    cv::Size ms;
    ms.height= temp_main_frame.size().height;
    ms.width = temp_main_frame.size().width;
    QSize qs = analytics_label->size();
    qdisplay_size = imageDisplaySize(ms,qs);

    QImage scaledImage = image.scaled(qdisplay_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    //std::cout<<ui->label_image->size().height()<<std::endl;

    if(scaledImage.isNull())
    {
        qDebug()<<"********Image is null!********";
    }
    else
    {
        analytics_label->setPixmap(QPixmap::fromImage(scaledImage));
    }
}
QSize MainWindow::imageDisplaySize(cv::Size &src_size,QSize &label_img_size)
{
    QSize qSize;
    int edge_px = 5;//this can avoid the faile of imageDisplay
    float y_scale = (float)(src_size.height)/(float)(label_img_size.height());
    float x_scale = (float)(src_size.width)/(float)(label_img_size.width());
    if(x_scale<y_scale)
    {
        qSize.setWidth(src_size.width/y_scale-edge_px);
        qSize.setHeight(src_size.height/y_scale-edge_px);
    }
    else
    {
        qSize.setWidth(src_size.width/x_scale-edge_px);
        qSize.setHeight(src_size.height/x_scale-edge_px);
    }
    return qSize;
}
void MainWindow::slot_close_video()
{

//    cv::Mat temp_main_frame;
//    cv::Mat blank_mat(h,w,CV_8UC3,cv::Scalar(255,255,255));
//    cv::cvtColor(blank_mat, temp_main_frame,CV_BGR2RGB);
//    QImage image = QImage((uchar*) temp_main_frame.data, temp_main_frame.cols, temp_main_frame.rows, temp_main_frame.step, QImage::Format_RGB888);
//    cv::Size ms;
//    ms.height= temp_main_frame.size().height;
//    ms.width = temp_main_frame.size().width;
//    QSize qs = ui->label_landmark->size();
//    qdisplay_size = imageDisplaySize(ms,qs);
//    QImage scaledImage = image.scaled(qdisplay_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
//    ui->label_landmark->setPixmap(QPixmap::fromImage(scaledImage));
//    ui->label_landmark->repaint();

    if(this->poseEstState==POSERUNNING)
        this->poseEstState = POSESTOPED;

    this->main_timer->stop();

    while(true)
    {
        if(processStop>=1)break;
    }

    analytics_label->clear();
    writer.release();
    cap.release();
    this->src = NO_SRC;
    connect(ui->pushButton_camera,SIGNAL(clicked()),this,SLOT(slot_open_video()));
    connect(ui->pushButton_video,SIGNAL(clicked()),this,SLOT(slot_open_video_file()));

    disconnect(analytics_label,SIGNAL(signal_press_position(int ,int)),this,SLOT(slot_press_position(int,int)));
}
void MainWindow::slot_exit()
{
    slot_close_video();

    exit(0);
}
//void MainWindow::cursorPosCallBack(int x,int y)
//{
//    ui->label_2->setText(tr("X=%1 Y=%2").arg(x).arg(y));
//}
void MainWindow::slot_press_position(int x,int y)
{

     float fx=x-(analytics_label->size().width()-qdisplay_size.width())/2;
     float fy=y-(analytics_label->size().height()-qdisplay_size.height())/2;

     float fw = qdisplay_size.width();
     float fh = qdisplay_size.height();

     fx = fx* w/fw;
     fy = fy* h/fh;

     mouse_position.x = fx;
     mouse_position.y = fy;

     if(x<-900)
     {
         ui->lcdNumber->display(0);
         ui->lcdNumber_2->display(0);
         ui->lcdNumber_3->display(0);
         ui->lcdNumber_4->display(0);
         ui->lcdNumber_5->display(0);
         //ui->lcdNumber_6->display(0);
         ui->label_emotion->setText("emotion");
         skeleton_show_one = false;
         face_show_one = false;

         eyesAttention = 0;
         mouthAttention = 0;
         faceAttention = 0;
         emotionStyle = 0;
     }

     if(face_show_one)
     {
         ui->lcdNumber->display(faceAttention);
         ui->lcdNumber_2->display(eyesAttention);
         ui->lcdNumber_3->display(mouthAttention);
         ui->label_emotion->setText(QString::fromStdString(emotion_des));
         //ui->lcdNumber_6->display(emotionStyle);
     }
     if(skeleton_show_one == true)
     {
        ui->lcdNumber_4->display(headAttention);
        ui->lcdNumber_5->display(shoulderAttention);
     }

}
std::vector<seeta::FaceInfo> MainWindow::faceDetection(cv::Mat img)
{
        cv::Mat img_gray;

        if (img.channels() != 1)
            cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        else
            img_gray = img;

        seeta::ImageData img_data;
        img_data.data = img_gray.data;
        img_data.width = img_gray.cols;
        img_data.height = img_gray.rows;
        img_data.num_channels = 1;

        std::vector<seeta::FaceInfo> faces = seetaRecognizer->Detect(img_data);
        //std::cout << "Image size (wxh): " << img_data.width << "x" << img_data.height << std::endl;
        cv::Rect face_rect;
        int32_t num_face = static_cast<int32_t>(faces.size());

        int nosePoint[1][2];

        std::vector<seeta::FaceInfo> faces1;//strategy 1;

        for (int32_t i = 0; i < num_face; i++) {
            face_rect.x = faces[i].bbox.x;
            face_rect.y = faces[i].bbox.y;
            face_rect.width = faces[i].bbox.width;
            face_rect.height = faces[i].bbox.height;
            if(face_showable)
                cv::rectangle(img, face_rect, CV_RGB(255, 255, 255), 2, 8, 0);


            nosePoint[0][0] = double(face_rect.x)+double(face_rect.width)/2.0;
            nosePoint[0][1] = double(face_rect.y)+double(face_rect.height)/2.0;
//            int mn_distance = mouse_nose_distance(nosePoint);
//            std::cout<<"faces1------------------------------------------"<<mn_distance<<std::endl;
//            if(mn_distance<mn_distance_thredshold-15)
//            {
//                std::cout<<"faces1------------------------------------------"<<std::endl;
//                faces1.resize(1);
//                faces1[0] = faces[i];
//            }
        }
        if(thread_strategy == 1)
            return faces1;
        else
            return faces;
}
void MainWindow::NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double> >& face_detections)
{
    // Go over the model and eliminate detections that are not informative (there already is a tracker there)
    for(size_t model = 0; model < clnf_models.size(); ++model)
    {

        // See if the detections intersect
        cv::Rect_<double> model_rect = clnf_models[model].GetBoundingBox();

        for(int detection = face_detections.size()-1; detection >=0; --detection)
        {
            double intersection_area = (model_rect & face_detections[detection]).area();
            double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

            // If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
            if( intersection_area/union_area > 0.5)
            {
                face_detections.erase(face_detections.begin() + detection);
            }
        }
    }
}
void MainWindow::transSeetaface2Rect(std::vector<FaceInfo> &seetaFaces, std::vector<cv::Rect_<double> > &face_detections)
{
    int add_field = 20;

    for(int i=0;i<seetaFaces.size();i++)
    {
        FaceInfo fi = seetaFaces[i];
        cv::Rect_<double> face_rect(fi.bbox.x-add_field,fi.bbox.y-add_field,fi.bbox.width+2*add_field,fi.bbox.height+2*add_field+20);
        if (face_rect.x <= 0)
            face_rect.x = 1;
        if (face_rect.y <= 0)
            face_rect.y = 1;
        if (face_rect.x >= w)
            face_rect.x = w-2;
        if (face_rect.y >= h)
            face_rect.y = h-2;
        if (face_rect.x + face_rect.width >= w)
            face_rect.width = w - face_rect.x - 1;
        if (face_rect.y + face_rect.height >= h)
            face_rect.height = h - face_rect.y - 1;
        if (face_rect.width <= 0)
            face_rect.width = 1;
        if (face_rect.height <= 0)
            face_rect.height = 1;
        face_detections.push_back(face_rect);
    }
}

void MainWindow::landmarkInit()
{
    vector<string> arguments ;
    arguments.push_back(string(getcwd(NULL, 0)));
    det_params = new LandmarkDetector::FaceModelParameters(arguments);

    det_params->use_face_template = true;
    // This is so that the model would not try re-initialising itself
    det_params->reinit_video_every = -1;

    det_params->curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;

    det_parameters.push_back(*det_params);

    clnf_model = new LandmarkDetector::CLNF(((LandmarkDetector::FaceModelParameters)det_parameters[0]).model_location);//time comsuming

    clnf_model->face_detector_HAAR.load(((LandmarkDetector::FaceModelParameters)det_parameters[0]).face_detector_location);
    clnf_model->face_detector_location = ((LandmarkDetector::FaceModelParameters)det_parameters[0]).face_detector_location;

    clnf_models.reserve(num_faces_max);

    clnf_models.push_back(*clnf_model);//time consuming

    active_models.push_back(false);
    emotion_predictions.resize(num_faces_max);
    for (int i = 1; i < num_faces_max; ++i)
    {
        std::cout<<"initial:"<<i<<std::endl;
        clnf_models.push_back(*clnf_model);//time consuming
        active_models.push_back(false);
        det_parameters.push_back(*det_params);

        emotion_predictions.push_back("neutral");
        features.push_back(new svm_node[272+1]);
        //std::cout<<"model:"<<i<<std::endl;
    }

    if(cx == 0 || cy == 0)
    {
        cx_undefined = true;
    }

    t0 = cv::getTickCount();

    svmModel = svm_load_model("model_pureCpp.txt");

    iniStep++;
}
int MainWindow::landmarkDetect(cv::Mat &captured_image, vector<cv::Rect_<double> > &face_detections)
{
    if(face_detections.size()==0)
        return 0;
    if(cx_undefined)
    {
        cx = captured_image.cols / 2.0f;
        cy = captured_image.rows / 2.0f;
    }

    cv::Mat_<float> depth_image;
    cv::Mat_<uchar> grayscale_image;

    cv::Mat disp_image = captured_image;

    cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);

    for(unsigned int model = 0; model < clnf_models.size(); ++model)
    {
        if(!active_models[model])
        {
            all_models_active = false;
        }
    }
    // Get the detections (every 8th frame and when there are free models available for tracking)
//    if(frame_count % 8 == 0 && !all_models_active)
//    {
//        std::vector<seeta::FaceInfo> faces = this->faceDetection(main_frame);
//        this->transSeetaface2Rect(faces,this->face_detections);
//    }
    //cout<<"this face detection size:"<<this->face_detections.size()<<endl;
    NonOverlapingDetections(clnf_models, face_detections);

    vector<tbb::atomic<bool> > face_detections_used(face_detections.size());

    tbb::parallel_for(0, (int)clnf_models.size(), [&](int model)
    {
        bool detection_success = false;

        // If the current model has failed more than 4 times in a row, remove it
        if(clnf_models[model].failures_in_a_row > 4)
        {
            active_models[model] = false;
            clnf_models[model].Reset();

            emotion_predictions[model] = "neutral";
            vector<svm_node*>().swap(features);

            for (int i = 1; i < num_faces_max; ++i)
                features.push_back(new svm_node[272+1]);

        }

        // If the model is inactive reactivate it with new detections
        if(!active_models[model])
        {

            for(size_t detection_ind = 0; detection_ind < face_detections.size(); ++detection_ind)
            {
                // if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
                if(face_detections_used[detection_ind].compare_and_swap(true, false) == false)
                {

                    // Reinitialise the model
                    clnf_models[model].Reset();

                    // This ensures that a wider window is used for the initial landmark localisation
                    clnf_models[model].detection_success = false;
                    detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, face_detections[detection_ind], clnf_models[model], det_parameters[model]);

                    // This activates the model
                    active_models[model] = true;

                    // break out of the loop as the tracker has been reinitialised
                    break;
                }

            }
        }
        else
        {
            // The actual facial landmark detection / tracking
            detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_models[model], det_parameters[model]);
        }
    });

    for(size_t model = 0; model < clnf_models.size(); ++model)
    {
        // Visualising the results
        // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
        double detection_certainty = clnf_models[model].detection_certainty;

        double visualisation_boundary = -0.1;

        // Only draw if the reliability is reasonable, the value is slightly ad-hoc
        if(detection_certainty < visualisation_boundary)
        {
            if(landmark_showable)
                LandmarkDetector::Draw(disp_image, clnf_models[model]);
            clnf_model[model].hierarchical_models;

            if(detection_certainty > 1)
                detection_certainty = 1;
            if(detection_certainty < -1)
                detection_certainty = -1;

            detection_certainty = (detection_certainty + 1)/(visualisation_boundary +1);

            // A rough heuristic for box around the face width
            int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

            // Work out the pose of the head from the tracked model
            cv::Vec6d pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(clnf_models[model], fx, fy, cx, cy);

            //std::cout<<"cols::::::"<<clnf_models[model].detected_landmarks.cols<<std::endl;
            //std::cout<<"rows::::::"<<clnf_models[model].detected_landmarks.rows<<std::endl;
            //cv::Mat_ <double> mats;
            cv::Point nose_point;
            nose_point.x = *(clnf_models[model].detected_landmarks[33]);
            nose_point.y = *(clnf_models[model].detected_landmarks[101]);

            //cv::ellipse(disp_image,nose_point,cv::Size(20,20),0,0,0,cv::Scalar(0,0,255),40,8);

            //cv::ellipse(disp_image,mouse_position,cv::Size(20,20),0,0,0,cv::Scalar(255,0,0),50,8);

            //std::cout<<"yigeyuansu--------"<<*(clnf_models[model].detected_landmarks[0])<<"--------diergeyuansu "<<*(clnf_models[model].detected_landmarks[18])<<std::endl;
            vector<cv::Point2d> faceLandMarks;
            faceLandMarks.resize(68);
            for(int markIndex = 0;markIndex<68;markIndex++)
            {
                faceLandMarks[markIndex].x = *(clnf_models[model].detected_landmarks[markIndex]);
                faceLandMarks[markIndex].y = *(clnf_models[model].detected_landmarks[markIndex+68]);
            }
            int nosePoint[1][2];
            nosePoint[0][0] = nose_point.x;
            nosePoint[0][1] = nose_point.y;
            int mn_distance = mouse_nose_distance(nosePoint);
            //std::cout<<"mouse_nose_distance(nosePoint):"<<mn_distance<<std::endl;

            // Draw it in reddish if uncertain, blueish if certain
            if(landmark_showable)
                LandmarkDetector::DrawLine(disp_image, pose_estimate, cv::Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);
                //LandmarkDetector::DrawBox(disp_image, pose_estimate, cv::Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);

            //landmarks2features(&clnf_models[model], features[model]);
//            int predictValue=svm_predict(svmModel, features[model]);
//            double class_nr = 0;
//            int class_nr_int = 0;
//            class_nr_int = (int)predictValue;
//            emotion_predictions[model] = result2labels(class_nr_int);

            std::cout<<"emotion_prediction::"<<emotion_predictions[model]<<std::endl;

            if(mn_distance<mn_distance_thredshold)
            {
//                mouth1.x = *(clnf_models[model].detected_landmarks[60]);
//                mouth1.y = *(clnf_models[model].detected_landmarks[128]);

//                cv::ellipse(disp_image,mouth1,cv::Size(20,20),0,0,0,cv::Scalar(255,0,0),20,8);

//                cv::Point mouth1,mouth2,mouth3,mouth4;
                //faceAttention = getFaceAttention();
                faceAttention = getFaceAttention(pose_estimate);
                eyesAttention = getEyesAttention(faceLandMarks);
                mouthAttention = getMouthAttention(faceLandMarks);
                //emotionStyle = class_nr_int;
                face_show_one = true;
            }

        }
    }

    // Work out the framerate
    if(frame_count % 10 == 0)
    {
        t1 = cv::getTickCount();
        fps = 10.0 / (double(t1-t0)/cv::getTickFrequency());
        t0 = t1;
    }

    // Write out the framerate on the image before displaying it
    char fpsC[255];
    sprintf(fpsC, "%d", (int)fps);
    string fpsSt("FPS:");
    fpsSt += fpsC;
    cv::putText(disp_image, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1, CV_AA);

    int num_active_models = 0;

    for( size_t active_model = 0; active_model < active_models.size(); active_model++)
    {
        if(active_models[active_model])
        {
            num_active_models++;
        }
    }

    char active_m_C[255];
    sprintf(active_m_C, "%d", num_active_models);
    string active_models_st("Active models:");
    active_models_st += active_m_C;
    cv::putText(disp_image, active_models_st, cv::Point(10,60), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1, CV_AA);

    // Update the frame count
    frame_count++;
}

int MainWindow::landmarkDetect1(cv::Mat &captured_image, vector<cv::Rect_<double> > &face_detections)
{
    if(face_detections.size()==0)
        return 0;
    if(!landmark_showable)
        return 0;
    if(cx_undefined)
    {
        cx = captured_image.cols / 2.0f;
        cy = captured_image.rows / 2.0f;
    }

    cv::Mat_<float> depth_image;
    cv::Mat_<uchar> grayscale_image;

    cv::Mat disp_image = captured_image;

    cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);

    for(unsigned int model = 0; model < clnf_models.size(); ++model)
    {
        if(!active_models[model])
        {
            all_models_active = false;
        }
    }
    // Get the detections (every 8th frame and when there are free models available for tracking)
//    if(frame_count % 8 == 0 && !all_models_active)
//    {
//        std::vector<seeta::FaceInfo> faces = this->faceDetection(main_frame);
//        this->transSeetaface2Rect(faces,this->face_detections);
//    }
    //cout<<"this face detection size:"<<this->face_detections.size()<<endl;
    NonOverlapingDetections(clnf_models, face_detections);

    vector<tbb::atomic<bool> > face_detections_used(face_detections.size());

    //bool detection_success = false;
    tbb::parallel_for(0, (int)clnf_models.size(), [&](int model)
    {
        bool detection_success = false;

        // If the current model has failed more than 4 times in a row, remove it
        if(clnf_models[model].failures_in_a_row > 4)
        {
            active_models[model] = false;
            clnf_models[model].Reset();

            //emotion_predictions[model] = "neutral";
            //vector<svm_node*>().swap(features);

//            for (int i = 1; i < num_faces_max; ++i)
//                features.push_back(new svm_node[272+1]);

        }
        // If the model is inactive reactivate it with new detections
        if(!active_models[model])
        {

            for(size_t detection_ind = 0; detection_ind < face_detections.size(); ++detection_ind)
            {
                // if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
                if(face_detections_used[detection_ind].compare_and_swap(true, false) == false)
                {

                    // Reinitialise the model
                    clnf_models[model].Reset();

                    // This ensures that a wider window is used for the initial landmark localisation
                    clnf_models[model].detection_success = false;
                    detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, face_detections[detection_ind], clnf_models[model], det_parameters[model]);

                    // This activates the model
                    active_models[model] = true;

                    // break out of the loop as the tracker has been reinitialised
                    break;
                }
            }
        }
        else
        {
            // The actual facial landmark detection / tracking
            detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_models[model], det_parameters[model]);
        }
    });
    cv::Mat faces_mat(h,w,CV_8UC3,cv::Scalar(255,255,255));

    for(size_t model = 0; model < clnf_models.size(); ++model)
    {
        // Visualising the results
        // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
        double detection_certainty = clnf_models[model].detection_certainty;

        double visualisation_boundary = -0.1;

        // Only draw if the reliability is reasonable, the value is slightly ad-hoc
        if(detection_certainty < visualisation_boundary)
        {
            if(landmark_showable){
                LandmarkDetector::Draw(disp_image, clnf_models[model]);
                LandmarkDetector::Draw(faces_mat, clnf_models[model]);
            }
            clnf_model[model].hierarchical_models;

            if(detection_certainty > 1)
                detection_certainty = 1;
            if(detection_certainty < -1)
                detection_certainty = -1;

            detection_certainty = (detection_certainty + 1)/(visualisation_boundary +1);

            // A rough heuristic for box around the face width
            int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

            // Work out the pose of the head from the tracked model
            cv::Vec6d pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(clnf_models[model], fx, fy, cx, cy);

            //std::cout<<"cols::::::"<<clnf_models[model].detected_landmarks.cols<<std::endl;
            //std::cout<<"rows::::::"<<clnf_models[model].detected_landmarks.rows<<std::endl;
            //cv::Mat_ <double> mats;
            cv::Point nose_point;
            nose_point.x = *(clnf_models[model].detected_landmarks[33]);
            nose_point.y = *(clnf_models[model].detected_landmarks[101]);

            //cv::ellipse(disp_image,nose_point,cv::Size(20,20),0,0,0,cv::Scalar(0,0,255),40,8);

            //cv::ellipse(disp_image,mouse_position,cv::Size(20,20),0,0,0,cv::Scalar(255,0,0),50,8);

            //std::cout<<"yigeyuansu--------"<<*(clnf_models[model].detected_landmarks[0])<<"--------diergeyuansu "<<*(clnf_models[model].detected_landmarks[18])<<std::endl;
            vector<cv::Point2d> faceLandMarks;
            faceLandMarks.resize(68);
            for(int markIndex = 0;markIndex<68;markIndex++)
            {
                faceLandMarks[markIndex].x = *(clnf_models[model].detected_landmarks[markIndex]);
                faceLandMarks[markIndex].y = *(clnf_models[model].detected_landmarks[markIndex+68]);
            }
            int nosePoint[1][2];
            nosePoint[0][0] = nose_point.x;
            nosePoint[0][1] = nose_point.y;
            int mn_distance = mouse_nose_distance(nosePoint);
            //std::cout<<"mouse_nose_distance(nosePoint):"<<mn_distance<<std::endl;

            // Draw it in reddish if uncertain, blueish if certain
            if(landmark_showable){
                //LandmarkDetector::DrawBox(disp_image, pose_estimate, cv::Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);
                //LandmarkDetector::DrawLine(disp_image, pose_estimate, cv::Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);
                //LandmarkDetector::DrawLine(faces_mat, pose_estimate, cv::Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);
            }
            //std::cout<<"model:"<<model<<std::endl;
            features[model] = new svm_node[272+1];
            features[model][0].index =0;
            landmarks2features(&clnf_models[model], features[model]);

            //svm_predict(svmModel, features[model]);

            int predictValue=svm_predict(svmModel, features[model]);

            delete features[model];

            double class_nr = 0;
            int class_nr_int = 0;
            class_nr_int = (int)predictValue;
            emotion_predictions[model] = result2labels(class_nr_int);

            //std::cout<<"emotion_prediction::"<<emotion_predictions[model]<<std::endl;

            if(mn_distance<mn_distance_thredshold)
            {
//                mouth1.x = *(clnf_models[model].detected_landmarks[60]);
//                mouth1.y = *(clnf_models[model].detected_landmarks[128]);

//                cv::ellipse(disp_image,mouth1,cv::Size(20,20),0,0,0,cv::Scalar(255,0,0),20,8);

//                cv::Point mouth1,mouth2,mouth3,mouth4;
                //faceAttention = getFaceAttention();
                faceAttention = getFaceAttention(pose_estimate);
                eyesAttention = getEyesAttention(faceLandMarks);
                mouthAttention = getMouthAttention(faceLandMarks);
                emotionStyle = class_nr_int;

                emotion_des = emotion_predictions[model];
                face_show_one = true;
            }

        }
    }

    // Work out the framerate
    if(frame_count % 10 == 0)
    {
        t1 = cv::getTickCount();
        fps = 10.0 / (double(t1-t0)/cv::getTickFrequency());
        t0 = t1;
    }

    // Write out the framerate on the image before displaying it
    char fpsC[255];
    sprintf(fpsC, "%d", (int)fps);
    string fpsSt("FPS:");
    fpsSt += fpsC;
    //cv::putText(disp_image, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1, CV_AA);

    int num_active_models = 0;

    for( size_t active_model = 0; active_model < active_models.size(); active_model++)
    {
        if(active_models[active_model])
        {
            num_active_models++;
        }
    }

    char active_m_C[255];
    sprintf(active_m_C, "%d", num_active_models);
    string active_models_st("Active models:");
    active_models_st += active_m_C;
    //cv::putText(disp_image, active_models_st, cv::Point(10,60), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1, CV_AA);

    // Update the frame count
    frame_count++;


    cv::Mat temp_main_frame;
    if(src != NO_SRC)
    {
        cv::cvtColor(faces_mat, temp_main_frame,CV_BGR2RGB);
    }
    else if(src == NO_SRC)
    {
        cv::Mat blank_mat(h,w,CV_8UC3,cv::Scalar(255,255,255));
        cv::cvtColor(blank_mat, temp_main_frame,CV_BGR2RGB);
    }

    QImage image = QImage((uchar*) temp_main_frame.data, temp_main_frame.cols, temp_main_frame.rows, temp_main_frame.step, QImage::Format_RGB888);
    cv::Size ms;
    ms.height= temp_main_frame.size().height;
    ms.width = temp_main_frame.size().width;
    QSize qs = ui->label_landmark->size();
    qdisplay_size = imageDisplaySize(ms,qs);
    scaledImage_landmark = image.scaled(qdisplay_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    //ui->label_landmark->setPixmap(QPixmap::fromImage(scaledImage_landmark));
    //ui->label_landmark->repaint();
    emit sig_paintLandmark();

}

void MainWindow::notify_face_ini_ok()
{
    QString tips("Face Ready!   Initializing...");
    iniReady(tips);
}

void MainWindow::poseInit()
{
    //realPose = new RTPose("/home/olsen305/rtpose_ying_li/caffe_rtpose/model/coco/pose_iter_440000.caffemodel", "/home/olsen305/rtpose_ying_li/caffe_rtpose/model/coco/pose_deploy_linevec.prototxt", 0);
//    if(this->poseEstState==POSESTOPED||this->poseEstState==POSEBORN)
//        this->poseEstState = POSEREADY;
    iniStep++;
}
void MainWindow::notify_pose_ini_ok()
{
    QString tips("Pose Ready!   Initializing...");
    iniReady(tips);
}
void MainWindow::iniReady(QString tips)
{
    statusBarLabel->setText(tips);
    if(iniStep == 2){
        statusBarLabel->setText("Ready!");
        ui->pushButton_camera->setDisabled(false);
        ui->pushButton_video->setDisabled(false);
        return;
    }
}
void MainWindow::runPoseEst()
{
    Json::Value root;
    Json::Reader reader;
    Json::FastWriter fastWriter;
    int count=0;

    int dataNum = 18;

    RTPose rt("/home/csuml/qilei_chen/caffe_rtpose/model/coco/pose_iter_440000.caffemodel", "/home/csuml/qilei_chen/caffe_rtpose/model/coco/pose_deploy_linevec.prototxt", 0);
    //std::string res_json = rt.getPoseEstimation(original_frame);
    while(this->poseEstState == POSERUNNING){
        cv::Mat skeleton_mat(h,w,CV_8UC3,cv::Scalar(255,255,255));
        if(this->poseImgQueue.size()>0)
        //if(this->poseImgQueue.size()>0&&(mouse_position.x>0))
        {
            //std::cout<<"mouse_position.x================"<<-1000<<std::endl;
            std::string res_json="null";
            if(skeleton_showable)
                res_json = rt.getPoseEstimation(poseImgQueue.front());
            //std::string res_json = realPose->getPoseEstimation(poseImgQueue.front());
            //std::cout<<res_json<<std::endl;
            reader.parse(res_json,root);
            //std::cout << "Output Json Parse result.\n";
            //std::cout << root["version"]<<std::endl;
            //std::cout <<"the 10th:" <<root["bodies"][10]<<std::endl;
            int i=0;
            std::string output = fastWriter.write(root["bodies"][i]);
            while(output.compare("null\n")!=0)
            {
                int joints_pos[joints_num][2];

                for(int j=0;j<joints_num;j++)
                {
                    std::string int_temp = fastWriter.write(root["bodies"][i]["joints"][j*3+0]);
                    joints_pos[j][0] = std::stoi(int_temp);
                    int_temp = fastWriter.write(root["bodies"][i]["joints"][j*3+1]);
                    joints_pos[j][1] = std::stoi(int_temp);
                    //std::cout<<joints_pos[j][0]<<"::::"<<joints_pos[j][1]<<std::endl;
                }
                if(skeleton_showable){
                    this->drawJoints(skeleton_mat,joints_pos);
                    this->drawBones(skeleton_mat,joints_pos);
                }
                skeleton_show_one = true;
                if(mouse_nose_distance(joints_pos)<mn_distance_thredshold)
                {
                headAttention = this->getAngleAttention(head_base,joints_pos,0,1);
                shoulderAttention = this->getAngleAttention(shoulder_base,joints_pos,5,2);
                }
                i++;
                output = fastWriter.write(root["bodies"][i]);
            }
            std::cout<<"pose number:"<<i<<"  count:"<<count++<<std::endl;
            //if(poseImgQueue.size()>1)
            //    poseImgQueue.pop();
        }
        cv::Mat temp_main_frame;
        cv::cvtColor(skeleton_mat, temp_main_frame,CV_BGR2RGB);
        //cv::imshow("te",skeleton_mat);
        //cv::waitKey(1);
        QImage image = QImage((uchar*) temp_main_frame.data, temp_main_frame.cols, temp_main_frame.rows, temp_main_frame.step, QImage::Format_RGB888);
        cv::Size ms;
        ms.height= temp_main_frame.size().height;
        ms.width = temp_main_frame.size().width;
        QSize qs = ui->skeleton_label->size();
        qdisplay_size = imageDisplaySize(ms,qs);
        scaledImage_skeleton = image.scaled(qdisplay_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        //ui->skeleton_label->setPixmap(QPixmap::fromImage(scaledImage_skeleton));
        //ui->skeleton_label->repaint();
        emit sig_paintSkeleton();
    }

    cv::Mat skeleton_mat(h,w,CV_8UC3,cv::Scalar(255,255,255));
    cv::Mat temp_main_frame;
    cv::cvtColor(skeleton_mat, temp_main_frame,CV_BGR2RGB);
    QImage image = QImage((uchar*) temp_main_frame.data, temp_main_frame.cols, temp_main_frame.rows, temp_main_frame.step, QImage::Format_RGB888);
    cv::Size ms;
    ms.height= temp_main_frame.size().height;
    ms.width = temp_main_frame.size().width;
    QSize qs = ui->skeleton_label->size();
    qdisplay_size = imageDisplaySize(ms,qs);
    scaledImage_skeleton = image.scaled(qdisplay_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    //ui->skeleton_label->setPixmap(QPixmap::fromImage(scaledImage_skeleton));
    //ui->skeleton_label->repaint();
    emit sig_paintSkeleton();
    //int t  = poseImgQueue.size();
    //for(int i=0;i<t;i++)
    //{
        //poseImgQueue.pop();
    //}
    rt.freeGPU();
    //realPose->freeGPU();
    //delete rt;
    //this->runPoseState = false;
    processStop++;
}
float MainWindow::mouse_nose_distance(int positions[][2])
{
    cv::Point nose_point;
    nose_point.x = positions[0][0];
    nose_point.y = positions[0][1];
    return cv::norm(nose_point-mouse_position);
}
void MainWindow::drawJoints(cv::Mat src,int positions[][2])
{
    for(int i=0;i<joints_num;i++)
    {
        cv::Point joint_point;
        joint_point.x  = positions[i][0];
        joint_point.y = positions[i][1];
        if(joint_point.x!=0)
            cv::ellipse(src,joint_point,cv::Size(20,20),0,0,0,cv::Scalar(0,0,255),20,8);
    }
}
void MainWindow::drawBones(cv::Mat src, int positions[][2])
{
    cv::Point start_point,end_point;

    to_point(start_point,end_point,positions,0,1);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,1,2);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,2,3);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,3,4);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,1,5);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,5,6);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,6,7);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,1,8);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,8,9);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,9,10);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,1,11);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,11,12);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,12,13);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,0,14);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,0,15);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,14,16);
    drawLine(src,start_point,end_point);

    to_point(start_point,end_point,positions,15,17);
    drawLine(src,start_point,end_point);

}
void MainWindow::to_point(cv::Point &start_point,cv::Point &end_point,int positions[][2],int start_index,int end_index)
{
    start_point.x = positions[start_index][0];
    start_point.y = positions[start_index][1];
    end_point.x = positions[end_index][0];
    end_point.y = positions[end_index][1];
}
void MainWindow::drawLine(cv::Mat src,cv::Point start_point,cv::Point end_point)
{
    if(start_point.x&&start_point.y&&end_point.x&&end_point.y)
    {
        cv::line(src,start_point,end_point,cv::Scalar(255,0,0),10,8);
    }
}

int MainWindow::getAngleAttention(int base_vector[2],int positions[][2],int index1,int index2)
{
    int reality_vector[2];
    if(positions[index1][0]&&positions[index1][1]&&positions[index2][1]&&positions[index2][1])
    {
        reality_vector[0] = positions[index1][0] - positions[index2][0];
        reality_vector[1] = positions[index1][1] - positions[index2][1];
        return angleAttention(base_vector,reality_vector);
    }
    else
        return 0;
}

int MainWindow::angleAttention(int base_vector[2],int reality_vector[2])
{

    double angle = calAngle(base_vector,reality_vector);
    if(angle>90)
        angle = 90;

    int attention = (1-angle/90)*100;
    return attention;
}

double MainWindow::calAngle(int a[2],int b[2])
{
    //double a[2]={0,100},b[2]={1,0};
    double ab,a1,b1,cosr;
    ab=double(a[0]*b[0]+a[1]*b[1]);
    a1=std::sqrt(double(a[0]*a[0]+a[1]*a[1]));
    b1=std::sqrt(double(b[0]*b[0]+b[1]*b[1]));
        cosr=ab/a1/b1;
    double angle = std::acos(cosr)*180/PI;
    return angle;
}

int MainWindow::getFaceAttention(cv::Vec6d &pose_estimate)
{
    double e1 = 100*std::fabs(pose_estimate[3]);
    double e2 = 100*std::fabs(pose_estimate[4]);
    double e3 = 100*std::fabs(pose_estimate[5]);
    return 100-(e1/3.0+e2/3.0+e3/3.0)*100/90.0;
}

int MainWindow::getEyesAttention(vector<cv::Point2d> faceLandMarks)
{
    //double d1 = two_points_distance(faceLandMarks,37,41);
    //double d2 = two_points_distance(faceLandMarks,36,39);
    double d1 = cv::norm(faceLandMarks[37]-faceLandMarks[41]);
    double d2 = cv::norm(faceLandMarks[36]-faceLandMarks[39]);
    //std::cout<<"eyes d1 d2:"<<d1<<"-------"<<d2<<std::endl;
    int Score = 200*(d1/d2);
    return Score;
}

int MainWindow::getMouthAttention(vector<cv::Point2d> faceLandMarks)
{
    //double d1 = two_points_distance(faceLandMarks,60,64);
    //double d2 = two_points_distance(faceLandMarks,62,66);
    double d1 = cv::norm(faceLandMarks[60]-faceLandMarks[64]);
    double d2 = cv::norm(faceLandMarks[62]-faceLandMarks[66]);
    //std::cout<<"Mouth d1 d2:"<<d1<<"-------"<<d2<<std::endl;
    int Score = 100*(1-d2/d1);
    return Score;
}

void MainWindow::slot_faces_check()
{
    this->face_showable = ui->check_face->isChecked();
}

void MainWindow::slot_emotion_check()
{
    this->emotion_showable = ui->check_emotion->isChecked();
}

void MainWindow::slot_skeleton_check()
{
    this->skeleton_showable = ui->check_skeleton->isChecked();
}

void MainWindow::slot_landmark_check()
{
    this->landmark_showable = ui->check_landmark->isChecked();
}

void MainWindow::landmarkDetect()
{
    //this->landmarkDetect(landmark_frame,face_detections);
    this->landmarkDetect1(landmark_frame,face_detections);

}
bool MainWindow::set_landmark_running(bool st)
{
    this->landmark_running = st;
}

double MainWindow::getDist(cv::Point2f p1, cv::Point2f p2)
{
    double r = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    return r;
}

void MainWindow::landmarks2features(LandmarkDetector::CLNF *clnf_model, svm_node *node)
{
    cv::Mat shape = clnf_model->detected_landmarks.clone();

    float ptx[68];
    float pty[68];
    float max_x = 0;
    float min_x = 65536;
    float max_y = 0;
    float min_y = 65536;
    float xmean = 0;
    float ymean = 0;
    float xsum = 0;
    float ysum = 0;

    for (int i = 0; i < 68; i++)
    {
        ptx[i] = shape.at<double>(i);
        pty[i] = shape.at<double>(i+68);
        if (ptx[i] > max_x){
            max_x = ptx[i];
        }
        if (ptx[i] < min_x){
            min_x = ptx[i];
        }
        if (pty[i] > max_y){
            max_y = pty[i];
        }
        if (pty[i] < min_y){
            min_y = pty[i];
        }
    }

    for (int i = 0; i < 68; i++)
    {
        ptx[i] = (ptx[i] - min_x)/(max_x - min_x);
        pty[i] = (pty[i] - min_y)/(max_y - min_y);
        cv::Point2f temp;
        temp.x = ptx[i];
        temp.y = pty[i];
        xsum += ptx[i];
        ysum += pty[i];
    }

    float xcentral[68];
    float ycentral[68];

    xmean = xsum/68.0;
    ymean = ysum/68.0;

    for (int i = 0; i < 68; i++)
    {
        xcentral[i] = ptx[i] - xmean;
        ycentral[i] = pty[i] - ymean;
    }
    int anglenose = 0;

    if ( ptx[26] == ptx[29] )
    {
        anglenose = 0;
    }
    else
    {
        anglenose = int(atan((pty[26]-pty[29])/(ptx[26]-ptx[29])) * 180 / PI);
    }
    if (anglenose < 0)
    {
        anglenose += 90;
    }
    else
    {
        anglenose -= 90;
    }

    // svm_node* node = new svm_node[272+1];

    int i = 0;

    for (int i = 0; i < 68; i++)
    {
        node[i * 4].index = i * 4;
        node[i * 4].value = xcentral[i];
        node[i * 4 + 1].index = i * 4 + 1;
        node[i * 4 + 1].value = ycentral[i];
        node[i * 4 + 2].index = i * 4 + 2;
        node[i * 4 + 2].value = getDist(cv::Point2f(xmean, ymean), cv::Point2f(ptx[i],pty[i]));
        node[i * 4 + 3].index = i * 4 + 3;
        node[i * 4 + 3].value = ((atan((pty[i] - ymean)/(ptx[i] - xmean))*180/PI) - anglenose)/180;
        //std::cout<<node[i*4].value<<std::endl;
    }
    node[272].index = -1;
}
string MainWindow::result2labels(int class_nr_int)
{
    string resultstr;

    if (class_nr_int == 0){
        resultstr = "neutral";
    }
    else if (class_nr_int == 1)
    {
        resultstr = "anger";
    }
    else if (class_nr_int == 2)
    {
        resultstr = "contempt";
    }
    else if (class_nr_int == 3)
    {
        resultstr = "disgust";
    }
    else if (class_nr_int == 4)
    {
        resultstr = "fear";
    }
    else if (class_nr_int == 5)
    {
        resultstr = "happy";
    }
    else if (class_nr_int == 6)
    {
        resultstr = "sadness";
    }
    else  if (class_nr_int == 7)
    {
        resultstr = "surprise";
    }
    else
    {
        std::cout << "i have no idea" << std::endl;
    }
    return resultstr;
}


void MainWindow::slot_paintSkeleton()
{
    ui->skeleton_label->setPixmap(QPixmap::fromImage(scaledImage_skeleton));
    ui->skeleton_label->repaint();
}

void MainWindow::slot_paintLandmark()
{
    ui->label_landmark->setPixmap(QPixmap::fromImage(scaledImage_landmark));
    ui->label_landmark->repaint();
}

//double MainWindow::two_points_distance(double marks[][2],int index1,int index2)
//{
//    double d=sqrt((marks[index1][0]-marks[index1][0])*(marks[index1][0]-marks[index2][0])+(marks[index1][1]-marks[index1][1])*(marks[index1][1]-marks[index2][1]));
//    return d;
//}
