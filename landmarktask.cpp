#include "landmarktask.h"
#include "mainwindow.h"

LandmarkTask::LandmarkTask(MainWindow *mw1)
{
    this->mw  = mw1;
}

void LandmarkTask::run()
{
    mw->landmarkDetect();
    mw->set_landmark_running(false);
}
