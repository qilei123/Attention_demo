#include <qrunnable.h>
#include <qthread.h>

class MainWindow;

#ifndef FACEINITASK_H
#define FACEINITASK_H

class FaceIniTask : public QThread
{
public:
    FaceIniTask(MainWindow *mw1);
    void run();
private:
    MainWindow *mw;
};

#endif // FACEINITASK_H
