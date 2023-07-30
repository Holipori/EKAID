# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'client_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(530, 583)
        self.ipEdit = QtWidgets.QLineEdit(Dialog)
        self.ipEdit.setGeometry(QtCore.QRect(110, 30, 113, 21))
        self.ipEdit.setObjectName("ipEdit")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 30, 60, 16))
        self.label.setObjectName("label")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(40, 370, 451, 191))
        self.textBrowser.setObjectName("textBrowser")
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(50, 70, 201, 201))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView_2.setGeometry(QtCore.QRect(280, 70, 201, 201))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.questionEdit = QtWidgets.QLineEdit(Dialog)
        self.questionEdit.setGeometry(QtCore.QRect(100, 340, 321, 21))
        self.questionEdit.setText("")
        self.questionEdit.setObjectName("questionEdit")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(40, 340, 60, 16))
        self.label_2.setObjectName("label_2")
        self.sendButton = QtWidgets.QPushButton(Dialog)
        self.sendButton.setGeometry(QtCore.QRect(420, 335, 71, 32))
        self.sendButton.setObjectName("sendButton")
        self.connectButton = QtWidgets.QPushButton(Dialog)
        self.connectButton.setGeometry(QtCore.QRect(230, 25, 81, 32))
        self.connectButton.setObjectName("connectButton")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(90, 280, 91, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(330, 280, 111, 16))
        self.label_4.setObjectName("label_4")
        self.refreshButton = QtWidgets.QPushButton(Dialog)
        self.refreshButton.setGeometry(QtCore.QRect(200, 300, 113, 32))
        self.refreshButton.setObjectName("refreshButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Server ip:"))
        self.label_2.setText(_translate("Dialog", "Question:"))
        self.sendButton.setText(_translate("Dialog", "Send"))
        self.connectButton.setText(_translate("Dialog", "Connect"))
        self.label_3.setText(_translate("Dialog", "Main (current)"))
        self.label_4.setText(_translate("Dialog", "Reference (past)"))
        self.refreshButton.setText(_translate("Dialog", "Refresh"))

