#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
	ui.setupUi(this);

	// menu handlers
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(close()));
	connect(ui.actionRandomGeneration, SIGNAL(triggered()), this, SLOT(onRandomGeneration()));
	connect(ui.actionGenerateTrainingData, SIGNAL(triggered()), this, SLOT(onGenerateTrainingData()));
	connect(ui.actionGeneratePredictedData, SIGNAL(triggered()), this, SLOT(onGeneratePredictedData()));

	// setup layouts
	glWidget = new GLWidget3D(this);
	this->setCentralWidget(glWidget);
}

void MainWindow::keyPressEvent(QKeyEvent* e) {
	glWidget->keyPressEvent(e);
}

void MainWindow::keyReleaseEvent(QKeyEvent* e) {
	glWidget->keyReleaseEvent(e);
}

void MainWindow::onRandomGeneration() {
	glWidget->renderManager.removeObjects();
	glWidget->tree.generateRandom();
	glWidget->tree.generateGeometry(&glWidget->renderManager, false);
	glWidget->updateGL();
}

void MainWindow::onGenerateTrainingData() {
	glWidget->generateTrainingData();
}

void MainWindow::onGeneratePredictedData() {
	glWidget->generatePredictedData();
}
