#ifndef ANALYTICSLABEL_H
#define ANALYTICSLABEL_H
#include <QLabel>
#include <QMouseEvent>
#include <QCursor>
#include <QWidget>

#define ARRORCURSOR_ID 1
#define CROSSCURSOR_ID 2
//namespace Al {
//class AnalyticsLabel;
//}
class AnalyticsLabel : public QLabel
{
    Q_OBJECT

public:
    AnalyticsLabel();

    ~AnalyticsLabel();
signals:
    void signal_press_position(int,int);
    void signal_reset_position(int,int);

protected:
    virtual void mouseMoveEvent(QMouseEvent * event);
    virtual void mousePressEvent(QMouseEvent * event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void enterEvent(QMouseEvent* e);
    virtual void leaveEvent(QMouseEvent* e);
private:
    QCursor cursor_shape = Qt::ArrowCursor;
    int cursor_id = ARRORCURSOR_ID;

};

#endif // ANALYTICSLABEL_H
