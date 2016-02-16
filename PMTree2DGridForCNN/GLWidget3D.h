#pragma once

#include "glew.h"
#include <QGLWidget>
#include <QMouseEvent>
#include <QKeyEvent>
#include "Camera.h"
#include "ShadowMapping.h"
#include "RenderManager.h"
#include <vector>
#include "PMTree2D.h"

class MainWindow;

class GLWidget3D : public QGLWidget {
public:
	MainWindow* mainWin;
	Camera camera;
	glm::vec3 light_dir;
	glm::mat4 light_mvpMatrix;
	RenderManager renderManager;
	QPoint lastPos;
	bool ctrlPressed;
	bool shiftPressed;
	bool altPressed;
	pmtree::PMTree2D tree;

public:
	GLWidget3D(MainWindow *parent);
	void generateTrainingData();
	void generatePredictedData();
	void render();
	void drawScene();

	void keyPressEvent(QKeyEvent* e);
	void keyReleaseEvent(QKeyEvent* e);

protected:
	void mousePressEvent(QMouseEvent* e);
	void mouseMoveEvent(QMouseEvent* e);
	void mouseReleaseEvent(QMouseEvent* e);
	void wheelEvent(QWheelEvent* e);
	void initializeGL();
	void resizeGL(int width, int height);
	void paintGL();
};

