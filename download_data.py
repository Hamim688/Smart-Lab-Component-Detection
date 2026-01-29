from roboflow import Roboflow
rf = Roboflow(api_key="Q43E1QNGZ2oz7feVlrbR")
project = rf.workspace("hamim-ixooj").project("deteksi_komponen_lengkap")
version = project.version(3)
dataset = version.download("yolov8")
                