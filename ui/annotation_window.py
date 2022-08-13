"""
This code is written by Ridvan Salih KUZU @DLR
ABOUT SCRIPT:
This script implements all user interface functionalities

"""
import sys
import glob
import shutil
import json
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from ui.gui import Ui_MainWindow
from PyQt5.QtCore import pyqtSlot
import threading
import albumentations as A
from general.custom_data_generator import InstanceGenerator
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from general.post_processing import threshold_mask, post_processing #,binary_opening_closing
from PIL.ImageQt import ImageQt
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
import os
logging.disable(logging.WARNING)



class MyForm(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui._button_input_dir.clicked.connect(self.push_input_on_click)
        self.ui._button_output_dir.clicked.connect(self.push_output_on_click)
        self.ui._button_model_dir.clicked.connect(self.push_model_on_click)
        self.ui._button_start.clicked.connect(self.push_start_on_click)

        self.ui._button_good.clicked.connect(self.push_good_on_click)
        self.ui._button_moderate.clicked.connect(self.push_moderate_on_click)
        self.ui._button_bad.clicked.connect(self.push_bad_on_click)
        self.ui._button_save.clicked.connect(self.save_on_click)
        self.ui._button_overwrite.clicked.connect(self.overwrite_on_click)

        self.ui._slider_smooth.sliderReleased.connect(self.slider_value_changed_event)
        self.ui._slider_remove_hole.sliderReleased.connect(self.slider_value_changed_event)

        self.ui._combo_box.activated.connect(self.combo_value_changed_event)

        self.ui._rb_contour_view.clicked.connect(self.predicted_display_changed_event)
        self.ui._rb_mask_view.clicked.connect(self.predicted_display_changed_event)

        self.log_path=os.path.dirname(os.path.realpath(__file__))+os.sep+'log.json'
        self.process_info = {}
        self.is_warm_up=False


    def init_at_warm_up(self):
        self.is_warm_up=True
        self._out_dir_dict = {}
        self._temp_input = None
        self._temp_mask_raw = None
        self._temp_mask_processed = None
        self._temp_overlap=None
        self._temp_contour_processed = None

        self._model = None

        self.GLOBAL_COUNTER = 0
        self.FILE_NUMBER = 0
        self.pd_input_table = pd.DataFrame(columns=["disp_file","in_file","temp_file", "mask_file","contour_file", "dest_im_file", "dest_mask_file","dest_contour_file"])

    @pyqtSlot()
    def push_input_on_click(self):
        in_dir = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select input folder including BMP images:', 'F:\\',QtWidgets.QFileDialog.ShowDirsOnly)
        self.ui._line_input_dir.setText(in_dir)

    @pyqtSlot()
    def predicted_display_changed_event(self):
        self.display_next()


    @pyqtSlot()
    def push_output_on_click(self):
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select output folder for generated masks:', 'F:\\',QtWidgets.QFileDialog.ShowDirsOnly)
        self.ui._line_output_dir.setText(out_dir)

    @pyqtSlot()
    def push_model_on_click(self):
        model_dir = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Tensorflow model for predictions:', 'F:\\',QtWidgets.QFileDialog.ShowDirsOnly)
        self.ui._line_model_dir.setText(model_dir)

    @pyqtSlot()
    def push_start_on_click(self):
        threading.Thread(target=self.prepare_start()).start()


    @pyqtSlot()
    def push_good_on_click(self):
        self.evaluate_next('good')

    @pyqtSlot()
    def push_moderate_on_click(self):
        self.evaluate_next('moderate')

    @pyqtSlot()
    def push_bad_on_click(self):
        self.evaluate_next('bad')

    @pyqtSlot()
    def save_on_click(self):
        self.move_to_selected_folder(False)

    @pyqtSlot()
    def overwrite_on_click(self):
        self.move_to_selected_folder(True)

    @pyqtSlot()
    def slider_value_changed_event(self):
        try:
            post_processing(self._temp_input, self._temp_mask_raw, self._temp_overlap, self._temp_mask_processed, self._temp_contour_processed,
                            self.ui._slider_remove_hole.value(), self.ui._slider_smooth.value())
        except Exception as e:
            print(e)
        self.display_next()

    @pyqtSlot()
    def combo_value_changed_event(self):
        try:
            self.GLOBAL_COUNTER = self.ui._combo_box.currentIndex() - 1
            self.evaluate_next()

        except Exception as e:
            print(e)
        self.display_next()



    def evaluate_next(self, evaluation=None):
        if not self.is_warm_up:
            message = "Please Start Process First!!"
            QMessageBox.warning(self, 'WARNING', message, QMessageBox.Ok)

        self.ui._slider_remove_hole.setValue(0)
        self.ui._slider_smooth.setValue(0)
        if evaluation is not None:
            self.assign_to_selected_folder(evaluation)
        self.update_progress()
        is_ongiong=self.predict_next()
        if is_ongiong: self.display_next()

    def prepare_start(self):
        print("\n\n\nPREPARATION OF THE SYSTEM MAY TAKE SOME MINUTES ON CPU MACHINES..\n\n\n")
        self.warmup_system()
        self.predict_next()
        self.display_next()

    def warmup_system(self):
        self.init_at_warm_up()
        in_dir = os.path.abspath(self.ui._line_input_dir.text())
        if not os.path.exists(in_dir):
            message = "Input Folder \"{}\" is NOT found!!".format(in_dir)
            QMessageBox.warning(self, 'WARNING', message, QMessageBox.Ok)

        else:
            for idx, file in enumerate(glob.glob(in_dir + os.sep + "*.bmp")):
                self.pd_input_table.loc[idx] = [os.path.basename(file),file, None, None, None,None,None,None]
                self.ui._combo_box.addItem(os.path.basename(file))

            self.FILE_NUMBER = len(glob.glob(in_dir + os.sep + "*.bmp"))

        out_dir = os.path.abspath(self.ui._line_output_dir.text())
        if not os.path.exists(out_dir):
            message = "The output folder is NOT found. \nCan we create it for you?"
            reply=QMessageBox.information(self, 'INFORMATION', message, QMessageBox.Yes, QMessageBox.No)
            if reply ==QMessageBox.Yes:
                try:
                    os.makedirs(out_dir)
                except OSError as e:
                    print(e)
                    message = "The output folder cannot be created. \n Please check the application permissions."
                    QMessageBox.warning(self, 'WARNING',message, QMessageBox.Ok)
            else:
                return

        self.create_subfolders()

        if self.FILE_NUMBER==0:
            message = "The input folder is empty. \nPlease select a folder including BMP images."
            QMessageBox.warning(self, 'WARNING', message, QMessageBox.Ok)
        else:
            try:
                self._model = load_model(os.path.abspath(self.ui._line_model_dir.text()),compile=False)
            except Exception as e:
                print(e)
                message = "The model file is not found. \nPlease select a folder including a TF model."
                QMessageBox.warning(self, 'WARNING', message, QMessageBox.Ok)

    def create_subfolders(self, overwrite=False):
        out_dir = self.ui._line_output_dir.text()
        out_dir = os.path.abspath(out_dir)

        try:
            self._out_dir_dict['good']={}
            self._out_dir_dict['moderate']={}
            self._out_dir_dict['bad'] = {}
            self._out_dir_dict['peak_detection'] = {}

            self._out_dir_dict['good']['image'] = out_dir + os.sep + 'good'+os.sep +'image'+os.sep
            self._out_dir_dict['good']['mask'] = out_dir + os.sep +'good'+ os.sep +'mask'+os.sep
            self._out_dir_dict['good']['contour'] = out_dir + os.sep + 'good' + os.sep + 'contour' + os.sep

            self._out_dir_dict['moderate']['image'] = out_dir + os.sep +'moderate'+ os.sep +'image'+os.sep
            self._out_dir_dict['moderate']['mask'] = out_dir + os.sep +'moderate'+ os.sep +'mask'+os.sep
            self._out_dir_dict['moderate']['contour'] = out_dir + os.sep + 'moderate' + os.sep + 'contour' + os.sep

            self._out_dir_dict['bad']['image'] = out_dir + os.sep +'bad'+ os.sep +'image'+os.sep
            self._out_dir_dict['bad'] ['mask']= out_dir + os.sep +'bad'+ os.sep +'mask'+os.sep
            self._out_dir_dict['bad']['contour'] = out_dir + os.sep + 'bad' + os.sep + 'contour' + os.sep

            self._out_dir_dict['peak_detection']['image'] = out_dir + os.sep +'peak_detection'+ os.sep +'image'+os.sep
            self._out_dir_dict['peak_detection']['mask'] = out_dir + os.sep +'peak_detection'+ os.sep +'mask'+os.sep
            self._out_dir_dict['peak_detection']['contour'] = out_dir + os.sep + 'peak_detection' + os.sep + 'contour' + os.sep

            self._out_dir_dict['peak_detection']['peak_detection'] = out_dir + os.sep +'peak_detection'+os.sep

            for key,subdir in self._out_dir_dict.items():
                for subsubdir in subdir.values():
                    if not os.path.exists(subsubdir):
                        os.makedirs(subsubdir)
                    elif overwrite and key != 'peak_detection':
                        shutil.rmtree(subsubdir)
                        os.makedirs(subsubdir)

        except OSError as e:
            print(e)
            message = "The output folder is protected. \n Please check the application permissions."
            QMessageBox.warning(self, 'WARNING', message, QMessageBox.Ok)

    def predict_next(self):
        try:
            if self.GLOBAL_COUNTER < self.FILE_NUMBER:
                INPUT_SHAPE = [512, 512, 3]
                OUTPUT_SHAPE = [512, 512, 1]

                TEST_TRANSFORMATION = A.Compose([
                    A.Resize(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                    A.Normalize(mean=0, std=0.5),
                ])

                in_file_name = self.pd_input_table.iloc[self.GLOBAL_COUNTER]['in_file']
                image, input_source, _ = next(InstanceGenerator(TEST_TRANSFORMATION, in_file_name).__iter__())



                pred_mask = self._model.predict(image).squeeze()
                pred_mask = threshold_mask(pred_mask)
                #pred_mask=binary_opening_closing(pred_mask)
                pred_mask=Image.fromarray((pred_mask * 255).astype(np.uint8))

                self._temp_input = self._out_dir_dict['peak_detection']['image'] + os.path.basename(input_source)
                self._temp_mask_raw = self._out_dir_dict['peak_detection']['mask'] + os.path.basename(input_source).replace('.bmp', '_raw_mask.bmp')
                self._temp_overlap = self._out_dir_dict['peak_detection']['mask'] + os.path.basename(input_source).replace('.bmp', '_overlap.bmp')
                self._temp_mask_processed = self._out_dir_dict['peak_detection']['mask'] + os.path.basename(input_source).replace('.bmp','_mask.bmp')
                self._temp_contour_processed = self._out_dir_dict['peak_detection']['contour'] + os.path.basename(input_source).replace('.bmp', '_contour.bmp')

                try:
                    shutil.copy(input_source,self._temp_input)
                except shutil.SameFileError:
                    pass
                pred_mask.save(self._temp_mask_raw)
                post_processing(self._temp_input, self._temp_mask_raw, self._temp_overlap, self._temp_mask_processed, self._temp_contour_processed,
                                self.ui._slider_remove_hole.value(), self.ui._slider_smooth.value())
                return True
            else:

                self.GLOBAL_COUNTER = 0
                message = "The process is complete. \nYou can start another process."
                reply = QMessageBox.information(self, 'INFORMATION', message, QMessageBox.Ok)
                return False

        except Exception as e:
            print(e)
            message = "An Error occured. \n {}".format(e.__str__())
            QMessageBox.critical(self, 'ERROR', message, QMessageBox.Ok)



    def display_next(self):
        try:
            if self.ui._rb_mask_view.isChecked():
                temp_predicted=self._temp_mask_processed
            elif self.ui._rb_contour_view.isChecked():
                temp_predicted = self._temp_contour_processed



            q_image = ImageQt(Image.open(self._temp_overlap)).copy()
            q_mask = ImageQt(Image.open(temp_predicted)).copy()

            pix = QtGui.QPixmap.fromImage(q_image)
            mix = QtGui.QPixmap.fromImage(q_mask)

            self.ui._label_input_image.setPixmap(pix)
            self.ui._label_input_image.adjustSize()
            self.ui._label_predicted_mask.setPixmap(mix)
            self.ui._label_predicted_mask.adjustSize()

            self.ui._label_input_name.setText(os.path.basename(self._temp_input))
            self.ui._label_mask_name.setText(os.path.basename(temp_predicted))
        except Exception as e:
            print(e)
            #message = "An error occured. \n {}".format(e.__str__())
            #QMessageBox.critical(self, 'ERROR', message, QMessageBox.Ok)
            return

    def assign_to_selected_folder(self, selected_dir):
        try:
            self.pd_input_table.iloc[self.GLOBAL_COUNTER]['temp_file'] = self._temp_input
            self.pd_input_table.iloc[self.GLOBAL_COUNTER]['mask_file'] = self._temp_mask_processed
            self.pd_input_table.iloc[self.GLOBAL_COUNTER]['contour_file'] = self._temp_contour_processed
            self.pd_input_table.iloc[self.GLOBAL_COUNTER]['dest_im_file'] = self._out_dir_dict[selected_dir]['image'] + os.path.basename(self._temp_input)
            self.pd_input_table.iloc[self.GLOBAL_COUNTER]['dest_mask_file'] = self._out_dir_dict[selected_dir]['mask'] + os.path.basename(self._temp_mask_processed)
            self.pd_input_table.iloc[self.GLOBAL_COUNTER]['dest_contour_file'] = self._out_dir_dict[selected_dir]['contour'] + os.path.basename(self._temp_contour_processed)
        except Exception as e:
            print(e)
            # message = "An Error occured. \n {}".format(e.__str__())
            # QMessageBox.critical(self, 'ERROR', message, QMessageBox.Ok)
            return

    def move_to_selected_folder(self, overwrite):
        try:
            if not self.is_warm_up:
                message = "Please Start the Process First!!"
                QMessageBox.warning(self, 'WARNING', message, QMessageBox.Ok)

            if overwrite:
                self.create_subfolders(overwrite=True)
            for idx, row_items in self.pd_input_table.iterrows():
                if row_items['dest_im_file'] is not None:
                    try:
                        shutil.copy(row_items['temp_file'], row_items['dest_im_file'])
                    except shutil.SameFileError:
                        pass
                    try:
                        shutil.copy(row_items['mask_file'], row_items['dest_mask_file'])
                    except shutil.SameFileError:
                        pass
                    try:
                        shutil.copy(row_items['contour_file'], row_items['dest_contour_file'])
                    except shutil.SameFileError:
                        pass

            message = "The evaluated files are saved."
            reply = QMessageBox.information(self, 'INFORMATION', message, QMessageBox.Ok)
        except Exception as e:
            print(e)
            #message = "An Error occured. \n {}".format(e.__str__())
            #QMessageBox.critical(self, 'ERROR', message, QMessageBox.Ok)
            return


    def update_progress(self):
        self.GLOBAL_COUNTER += 1
        percentage = int(100 * self.GLOBAL_COUNTER/self.FILE_NUMBER)
        self.ui._progress_bar.setValue(percentage)
        self.ui._label_progress.setText('{} of {} files completed!'.format(self.GLOBAL_COUNTER,self.FILE_NUMBER))

    def prepare_closing(self):
        self.process_info['input_dir'] = self.ui._line_input_dir.text()
        self.process_info['output_dir'] = self.ui._line_output_dir.text()
        self.process_info['model_dir'] = self.ui._line_model_dir.text()
        with open(self.log_path, 'w') as outfile:
            json.dump(self.process_info, outfile)
        shutil.rmtree(self._out_dir_dict['peak_detection']['peak_detection'])

    def prepare_opening(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as readfile:
                self.process_info = json.load(readfile)
                self.ui._line_input_dir.setText(self.process_info['input_dir'])
                self.ui._line_output_dir.setText(self.process_info['output_dir'])
                self.ui._line_model_dir.setText(self.process_info['model_dir'])


if __name__ == "__main__":
    try:
        app = QtWidgets.QApplication(sys.argv)
        myapp = MyForm()
        myapp.prepare_opening()
        myapp.show()
        ret=app.exec_()
        myapp.prepare_closing()
        sys.exit(ret)
    except Exception as e:
        print(e)
