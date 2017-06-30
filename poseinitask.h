#include <qthread.h>

#ifndef POSEINITASK_H
#define POSEINITASK_H

class MainWindow;

class PoseIniTask:public QThread
{
public:
    PoseIniTask(MainWindow *mw1);
    void run();
private:
    MainWindow *mw;
};

#endif // POSEINITASK_H
