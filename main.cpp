#include "mainwindow.h"
#include <QApplication>

#include <pthread.h>

int main(int argc, char *argv[])
{
    //pthread_t tid;
    QApplication am(argc, argv);
    MainWindow w;
    w.show();
    //pthread_create(&tid, NULL, testrtpose, NULL);
    //testrtpose();
    //return 0;
    return am.exec();
}

