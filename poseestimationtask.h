#include<QThread>
#ifndef POSEESTIMATIONTASK_H
#define POSEESTIMATIONTASK_H

class MainWindow;

class PoseEstimationTask : public QThread
{
public:
    PoseEstimationTask(MainWindow *mw1);
    void run();
private:
    MainWindow *mw;
};

#endif // POSEESTIMATIONTASK_H
