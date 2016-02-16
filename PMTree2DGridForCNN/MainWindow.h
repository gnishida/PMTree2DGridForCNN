#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"
#include "GLWidget3D.h"

class MainWindow : public QMainWindow {
	Q_OBJECT

private:
	Ui::MainWindowClass ui;
	GLWidget3D* glWidget;

public:
	MainWindow(QWidget *parent = 0);

protected:
	void keyPressEvent(QKeyEvent* e);
	void keyReleaseEvent(QKeyEvent* e);

public slots:
	void onRandomGeneration();
	void onGenerateTrainingData();
	void onGeneratePredictedData();
};

#endif // MAINWINDOW_H
