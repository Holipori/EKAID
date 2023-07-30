# Import socket module
import socket
from tqdm import tqdm
import sys
from client_ui import Ui_Dialog
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5 import QtGui
import os
import threading
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap

class MyMainForm(QMainWindow, Ui_Dialog):
    mainSignal = pyqtSignal(str)
    '''
    GUI class. this is the code mainly taking charge of the GUI
    '''
    def __init__(self):
        super(MyMainForm, self).__init__()
        self.setupUi(self)
        self.s = None # initialize the socket object
        self.writeText('Please enter your server ip...')
        self.ipEdit.setText('goldmonkeyuta.uta.edu')
        self.connectButton.clicked.connect(lambda: self.new_therad())
        self.sendButton.clicked.connect(lambda: self.asking())
        self.refreshButton.clicked.connect(lambda: self.refreshing())
        self.sendButton.setEnabled(False)
        self.refreshButton.setEnabled(False)
        self.port = 4000
        self.signals = pyqtSignal(str)
    def new_therad(self):
        '''create a new thread to listen for incoming connection'''

        self.work = MyThread(self.ipEdit.text()) # new thread object
        self.work.writeSignals.connect(self.writeText)
        self.work.commandSignals.connect(self.processCommand)
        self.work.start()
        self.mainSignal.connect(self.work.on_message_from_main)
    def writeText(self, text):
        '''this function is for displaying to the text browser'''
        self.textBrowser.insertPlainText(text+'\n')
        self.textBrowser.moveCursor(QtGui.QTextCursor.End)
    def processCommand(self, command):
        command = command.split(',')
        if command[0] == 'clear':
            self.textBrowser.clear()
        if command[0] == 'button':
            if command[2] == 'True':
                if command[1] == 'sendButton':
                    self.sendButton.setEnabled(True)
                elif command[1] == 'refreshButton':
                    self.refreshButton.setEnabled(True)
            else:
                if command[1] == 'sendButton':
                    self.sendButton.setEnabled(False)
                elif command[1] == 'refreshButton':
                    self.refreshButton.setEnabled(False)
        if command[0] == 'load_image':
            main_image_path = command[1]
            ref_image_path = command[2]
            # load and rescale the image
            main_image = QPixmap(main_image_path)
            ref_image = QPixmap(ref_image_path)
            main_image = main_image.scaled(self.graphicsView.width(), self.graphicsView.height(), Qt.KeepAspectRatio)
            ref_image = ref_image.scaled(self.graphicsView_2.width(), self.graphicsView_2.height(), Qt.KeepAspectRatio)
            main_scene = QGraphicsScene()
            ref_scene = QGraphicsScene()
            main_scene.addPixmap(main_image)
            ref_scene.addPixmap(ref_image)
            self.graphicsView.setScene(main_scene)
            self.graphicsView_2.setScene(ref_scene)
            
            

    def asking(self):
        self.work.asked = True
        self.mainSignal.emit(self.questionEdit.text())
    def refreshing(self):
        self.work.refresh()



class MyThread(QThread):
    writeSignals = pyqtSignal(str)  # define signals to be string. signals will be used to communicate with the main thread.
    commandSignals = pyqtSignal(str)
    def __init__(self, ip):
        super(MyThread, self).__init__()
        self.ip = ip
        self.port = 4000
    def on_message_from_main(self, message):
        # self.asked = True
        self.question = message
    def refresh(self):
        self.s.send('refresh'.encode())
        self.commandSignals.emit('clear')
        self.load_image()
    def accept_file(self, file_size, file_name):
        # Check if the file size is an error message
        if file_size == -1:
            # Print the error message
            print('File not found')
        else:
            # Create a new file to write the data
            f = open('cache/' + file_name, 'wb')
            # Initialize a variable to store the received bytes
            received_bytes = 0
            # A loop to receive data until the file size is reached
            while received_bytes < file_size:
                # Receive data from the server
                data = self.s.recv(1024)
                # Write data to the file
                f.write(data)
                # Update the received bytes
                received_bytes += len(data)
            # Close the file
            f.close()
            # Print a success message
            print('File received successfully')
    def accept_image(self):
        response = self.s.recv(1024).decode()
        response = response.split(',')
        file_size = int(response[0])
        file_name = response[1]
        self.s.send('T'.encode())
        self.accept_file(file_size, file_name)
        self.s.send('T'.encode())
        return file_name
    def load_image(self):
        self.s.send('load_image'.encode())
        # Receive the file size from the server
        main_file_name = self.accept_image()
        ref_file_name = self.accept_image()
        subject_id = self.s.recv(1024).decode()
        main_file_path = 'cache/' + main_file_name
        ref_file_path = 'cache/' + ref_file_name
        self.commandSignals.emit('load_image,' + main_file_path + ',' + ref_file_path)
        self.subject_id = subject_id
        self.writeSignals.emit('Subject ID: ' + self.subject_id)




    def run(self,):
        self.asked = False
        # Create a socket object
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # clean cache folder
        for file in os.listdir('cache'):
            os.remove('cache/' + file)
        self.s.connect((self.ip, self.port))
        # Connect to the server on local computer
        # try:
        QApplication.processEvents()
        self.commandSignals.emit('clear')
        self.writeSignals.emit('Connected to the server successfully')
        self.load_image()
        self.commandSignals.emit('button,sendButton,True')
        self.commandSignals.emit('button,refreshButton,True')
        self.writeSignals.emit('Please enter your question...')
        # except:
        #     # self.textBrowser.clear()
        #     self.writeSignals.emit('Cannot connect to the server. Please check your ip address and try again.')
        #     return
        # A loop to send file names and receive files until exit()
        while True:
            start_time = time.time()
            while self.asked != True:
                QApplication.processEvents()
                time.sleep(0.05)
                # if time.time() - start_time > 20:
                #     self.writeText('timeout. please enter your question')  ## set a waiting time, just in case the unexpected process termination.

            # Input the file name from the user
            question = self.question
            self.writeSignals.emit('Q: ' + question)
            self.asked = False
            self.s.send('question'.encode()) # send the "question" command
            confirmation = self.s.recv(1024).decode() # receive the response
            if confirmation == 'ready':
                self.s.send(question.encode())
                answer = self.s.recv(1024).decode()
                if answer == 'error':
                    self.s.send('T'.encode())
                    example_questions = self.s.recv(1500).decode()
                    self.writeSignals.emit('Please enter a valid question. Here are some examples:')
                    self.writeSignals.emit(example_questions)
                    continue
                self.writeSignals.emit('A: ' + answer + '\n')
                # time.sleep(3)
                # self.writeSignals.emit('\nPlease enter another question or refresh')
            else:
                self.writeSignals.emit('server not ready')
                continue


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())

