#include "faceinitask.h"
#include "mainwindow.h"

FaceIniTask::FaceIniTask(MainWindow *mw1)
{
    mw = mw1;
}

void FaceIniTask::run()
{
    mw->landmarkInit();
    mw->notify_face_ini_ok();
}
