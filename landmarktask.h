#include <qthread.h>

class MainWindow;

#ifndef LANDMARKTASK_H
#define LANDMARKTASK_H


class LandmarkTask : public QThread
{
public:
    LandmarkTask(MainWindow *mw1);
    void run();
private:
    MainWindow *mw;

};

#endif // LANDMARKTASK_H
