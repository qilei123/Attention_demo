#include "analyticslabel.h"

AnalyticsLabel::AnalyticsLabel()
{
    this->setAlignment(Qt::AlignCenter);
    //this->setMouseTracking(true);
}

AnalyticsLabel::~AnalyticsLabel()
{

}
void AnalyticsLabel::mouseMoveEvent(QMouseEvent * event)
{
    this->setCursor(Qt::CrossCursor);
    //mainWindow->cursorPosCallBack(event->pos().x(),event->pos().y());
    //static_cast<QLabel*>(widget)->setText(tr("X=%1 Y=%2").arg(event->pos().x()).arg(event->pos().y()));
    emit signal_press_position(event->pos().x(),event->pos().y());
}


void AnalyticsLabel::mousePressEvent(QMouseEvent *event)
{

    if(cursor_id == ARRORCURSOR_ID)
    {
        cursor_id = CROSSCURSOR_ID;
        cursor_shape = Qt::CrossCursor;
    }
    else if(cursor_id = CROSSCURSOR_ID)
    {
        cursor_id = ARRORCURSOR_ID;
        cursor_shape = Qt::ArrowCursor;
    }
    this->setCursor(cursor_shape);

    emit signal_press_position(event->pos().x(),event->pos().y());

}
void AnalyticsLabel::mouseReleaseEvent(QMouseEvent *event)
{
    cursor_id = ARRORCURSOR_ID;
    this->setCursor(Qt::ArrowCursor);
    //emit signal_reset_position(-1000,-1000);
}
void AnalyticsLabel::enterEvent(QMouseEvent* event)
{

}
void AnalyticsLabel::leaveEvent(QMouseEvent* event)
{
    cursor_id = ARRORCURSOR_ID;
    this->setCursor(Qt::ArrowCursor);
    emit signal_reset_position(-1000,-1000);
}
//void AnalyticsLabel::signal_press_position(int x,int y)
//{

//}
