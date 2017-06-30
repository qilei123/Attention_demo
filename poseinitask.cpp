#include "poseinitask.h"
#include "mainwindow.h"

PoseIniTask::PoseIniTask(MainWindow *mw1)
{
    mw = mw1;
}

void PoseIniTask::run()
{
    mw->poseInit();
    mw->notify_pose_ini_ok();
}
