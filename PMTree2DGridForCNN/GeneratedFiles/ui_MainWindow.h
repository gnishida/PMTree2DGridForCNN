/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindowClass
{
public:
    QAction *actionExit;
    QAction *actionRandomGeneration;
    QAction *actionGenerateTrainingData;
    QAction *actionGenerateLocalTrainingData;
    QAction *actionGeneratePredictedData;
    QAction *actionGenerateTrainingDataTrunk;
    QAction *actionGeneratePredictedDataTrunk;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuPM;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindowClass)
    {
        if (MainWindowClass->objectName().isEmpty())
            MainWindowClass->setObjectName(QStringLiteral("MainWindowClass"));
        MainWindowClass->resize(751, 731);
        actionExit = new QAction(MainWindowClass);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        actionRandomGeneration = new QAction(MainWindowClass);
        actionRandomGeneration->setObjectName(QStringLiteral("actionRandomGeneration"));
        actionGenerateTrainingData = new QAction(MainWindowClass);
        actionGenerateTrainingData->setObjectName(QStringLiteral("actionGenerateTrainingData"));
        actionGenerateLocalTrainingData = new QAction(MainWindowClass);
        actionGenerateLocalTrainingData->setObjectName(QStringLiteral("actionGenerateLocalTrainingData"));
        actionGeneratePredictedData = new QAction(MainWindowClass);
        actionGeneratePredictedData->setObjectName(QStringLiteral("actionGeneratePredictedData"));
        actionGenerateTrainingDataTrunk = new QAction(MainWindowClass);
        actionGenerateTrainingDataTrunk->setObjectName(QStringLiteral("actionGenerateTrainingDataTrunk"));
        actionGeneratePredictedDataTrunk = new QAction(MainWindowClass);
        actionGeneratePredictedDataTrunk->setObjectName(QStringLiteral("actionGeneratePredictedDataTrunk"));
        centralWidget = new QWidget(MainWindowClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        MainWindowClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindowClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 751, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuPM = new QMenu(menuBar);
        menuPM->setObjectName(QStringLiteral("menuPM"));
        MainWindowClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindowClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindowClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindowClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindowClass->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuPM->menuAction());
        menuFile->addAction(actionExit);
        menuPM->addAction(actionRandomGeneration);
        menuPM->addSeparator();
        menuPM->addAction(actionGenerateTrainingData);
        menuPM->addSeparator();
        menuPM->addAction(actionGeneratePredictedData);

        retranslateUi(MainWindowClass);

        QMetaObject::connectSlotsByName(MainWindowClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindowClass)
    {
        MainWindowClass->setWindowTitle(QApplication::translate("MainWindowClass", "MainWindow", 0));
        actionExit->setText(QApplication::translate("MainWindowClass", "Exit", 0));
        actionRandomGeneration->setText(QApplication::translate("MainWindowClass", "Random Generation", 0));
        actionRandomGeneration->setShortcut(QApplication::translate("MainWindowClass", "Ctrl+G", 0));
        actionGenerateTrainingData->setText(QApplication::translate("MainWindowClass", "Generate Training Data", 0));
        actionGenerateLocalTrainingData->setText(QApplication::translate("MainWindowClass", "Generate Local Training Data", 0));
        actionGeneratePredictedData->setText(QApplication::translate("MainWindowClass", "Generate Predicted Data", 0));
        actionGenerateTrainingDataTrunk->setText(QApplication::translate("MainWindowClass", "Generate Training Data (Trunk)", 0));
        actionGeneratePredictedDataTrunk->setText(QApplication::translate("MainWindowClass", "Generate Predicted Data (Trunk)", 0));
        menuFile->setTitle(QApplication::translate("MainWindowClass", "File", 0));
        menuPM->setTitle(QApplication::translate("MainWindowClass", "PM", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindowClass: public Ui_MainWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
