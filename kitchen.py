from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.image import Image
import os, sys
import cv2
import time
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import userpaths
import json
import pickle
import face_recognition
from kivy.core.window import Window
from kivymd.uix.list import OneLineIconListItem, IconLeftWidget
import threading
from numba import jit
Window.maximize()

queue = list()
available = False
# Usage example of MDBackdrop.
class KivyCamera(Image):
    def __init__(self, **kwargs):
        path = userpaths.get_my_documents()+"\\attendance_APP"
        if os.path.isfile(path+ "\\app_data.json"):
            with open(path + "\\app_data.json", "r")as f :
                self.data = json.load(f)

        else:
            if not os.path.exists(path):
                os.makedirs(path)
            self.data = {
                "camera_addr":0,
                "FBS":30,
                "catching people frequency": 2,
                "image analysis frequency":0.5,
                "attendance list frequency":0.5
                }
            with open(path + "\\app_data.json", "w")as f :
                json.dump(self.data, f)


        self.faces = []
        self.frame = []

        super(KivyCamera, self).__init__(**kwargs)

        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.capture = cv2.VideoCapture(self.data["camera_addr"], cv2.CAP_DSHOW)

        Clock.schedule_interval(self.update, 1.0 / self.data["FBS"])
        Clock.schedule_interval(self.catch, 1.0 / self.data["catching people frequency"])

    def update(self, dt, *args):
        ret, self.frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Draw a rectangle around the faces
            for (x, y, w, h) in self.faces:
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # convert it to texture
            buf1 = cv2.flip(self.frame,
             0)
            buf = buf1.tobytes()
            image_texture = Texture.create(
                size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture

    def close(self, *args):
        self.capture.release()

    def open(self, *args):
        self.capture = cv2.VideoCapture(self.data["camera_addr"], cv2.CAP_DSHOW)


    def catch(self, *args):
        global available, queue
        img = self.frame
        face = self.faces
        if len(face) > 0:
            queue.append((img, face))
            available = True
            face = 0
            self.faces=0
        else:
            pass




class Example(MDApp):
    def detect(self, *args):

        global available, queue
        if available:
            here = queue
            queue = []
            available = False
            print(len(here))
            for data in here:
                imgs, faces = data
                faces_encodings = face_recognition.face_encodings(imgs, known_face_locations=faces)
                time.sleep(1.5)
                closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
                are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(faces))]
                predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(faces_encodings), faces, are_matches)]
                for name, (top, right, bottom, left) in predictions:
                    self.detected.add(name)
        threading.Timer(3, self.detect).start()



    def camera_switch_change(self, switch_obj, switch_val):
        if switch_val:
            self.my_camera.open()
            self.root.ids.camera_label.text = "camera is on!"

        else:
            self.my_camera.close()
            self.root.ids.camera_label.text = "camera is off!"




    def close_camera(self, *args):
        self.my_camera.close()
    def open_camera(self, *args):
        self.my_camera.open()
    def update_attendance(self, *args):
        for attendee in self.detected :
            if attendee not in self.attendance:

                self.attendance.append(attendee)
                icon =  IconLeftWidget ( icon = "checkbox-marked-circle")
                item = OneLineIconListItem(text=attendee)
                item.add_widget(icon)
                self.root.ids.container.add_widget(item)


    def on_start(self, *args):
        with open("trained_knn_model.clf", 'rb') as f:
            self.knn_clf = pickle.load(f)
        Clock.schedule_interval(self.update_attendance, 1.0 / self.my_camera.data["attendance list frequency"])
        self.detect()


    def build(self):
        self.detected = set()
        self.attendance = []
        self.my_camera = KivyCamera()
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Orange"

        return Builder.load_string(
        """
<DrawerClickableItem@MDNavigationDrawerItem>
    focus_color: "#e7e4c0"
    text_color: "#4a4939"
    icon_color: "#4a4939"
    ripple_color: "#c5bdd2"
    selected_color: "#0c6c4d"


<DrawerLabelItem@MDNavigationDrawerItem>
    text_color: "#4a4939"
    icon_color: "#4a4939"
    focus_behavior: False
    selected_color: "#4a4939"
    _no_ripple_effect: True

MDNavigationLayout:

    MDScreenManager:
        id: screen_manager
        Screen:
            name:"main"
            MDSwitch:
                id: camera_switch
                active: True
                on_active: app.camera_switch_change(self, self.active)

                # giving position to the switch on screen
                pos_hint: {'center_x': .36, 'center_y': .2}
            MDLabel:
                id: camera_label
                font_size: 32
                text : "camera is on!"
                pos_hint: {'center_x': .6, 'center_y': .2}

            MDCard:
                pos_hint: {"center_x": .27, "center_y": .7}
                size_hint: 0.6, 0.8
                KivyCamera:
                    size_hint: 1, 1
            MDCard:
                id:parent
                pos_hint: {"center_x": .8, "center_y": .48}
                size_hint: 0.35, 0.8
                MDScrollView:
                    MDList:
                        id: container




        Screen:
            name:"attendance"
            MDLabel:
                text:"attendance screen"
                halign: 'center'
        Screen:
            name:"setining"
            MDLabel:
                text:"setining screen"
                halign: 'center'



    MDTopAppBar:
        title: "Navigation Drawer"
        elevation: 10
        pos_hint: {"top": 1}
        md_bg_color: "#e7e4c0"
        specific_text_color: "#4a4939"
        left_action_items:
            [['menu', lambda x: nav_drawer.set_state("open")]]

    MDNavigationDrawer:
        id: nav_drawer
        radius: (0, 16, 16, 0)

        MDNavigationDrawerMenu:

            MDNavigationDrawerHeader:
                title: "options"
                title_color: "#4a4939"
                text: "sptions selection"
                spacing: "4dp"
                padding: "12dp", 0, 0, "56dp"

            MDNavigationDrawerLabel:
                text: "main"

            DrawerClickableItem:
                icon: "camera"
                text_right_color: "#4a4939"
                text: "main screen"
                on_release:
                    screen_manager.current = "main"

            DrawerClickableItem:
                icon: "account-check"
                text: "attendance list"
                on_release:
                    screen_manager.current = "attendance"

            MDNavigationDrawerDivider:
            MDNavigationDrawerLabel:
                text: "setining"


            DrawerClickableItem:
                icon: "cog"
                text: "setining"
                on_release:
                    screen_manager.current = "setining"


        """

        )


if __name__ == '__main__':
    if hasattr(sys, '_MEIPASS'):
        resource_add_path(os.path.join(sys._MEIPASS))
    Example().run()



"""


                    """
