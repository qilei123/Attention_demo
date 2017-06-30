#include "poseestimationtask.h"

#include "mainwindow.h"

PoseEstimationTask::PoseEstimationTask(MainWindow *mw1)
{
    this->mw = mw1;
}

void PoseEstimationTask::run()
{
    this->mw->runPoseEst();
}
