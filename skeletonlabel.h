#ifndef SKELETONLABEL_H
#define SKELETONLABEL_H
#include <QLabel>
#include <QPainter>
#include <QPen>

class SkeletonLabel : public QLabel
{
Q_OBJECT
public:
    SkeletonLabel();
    void paintEvent(QPaintEvent *event);
};

#endif // SKELETONLABEL_H
