#include "skeletonlabel.h"

SkeletonLabel::SkeletonLabel()
{

}

void SkeletonLabel::paintEvent(QPaintEvent *event)
{
    QLabel::paintEvent(event);
    QPainter painter(this);
    QPen pen;
    pen.setWidth(Qt::red);
    pen.setWidth(2);
    painter.setPen(pen);
    painter.drawLine(1,1,20,20);
}
