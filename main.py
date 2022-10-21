import os
# from detectron2.engine import DefaultPredictor
# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# import cv2
from app import app
import urllib.request
import uuid
from flask import Flask, flash, abort, jsonify, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

@app.route("/route")
def hello_world(filename):
    return 
    #config = setup_detectron()
    config = 'config'
    vid_path = request.args.get("vid", default="test.mp4", type=str)
    preds = predict(config, vid_path)
    return {
            "words": preds
        }

def setup_detectron():
    zoo_config = model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )
    config = get_cfg() 
    config.merge_from_file(zoo_config)
    config.MODEL.ROI_HEADS.NUM_CLASSES = 2
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    config.MODEL.WEIGHTS = "models/model_final.pth"
    config.MODEL.DEVICE = "cpu"
    return config

def mapper(id: int) -> str:
    d = {
        0: "hello",
        1: "yes"
    }
    return d[id]

def predict(config, vid_path: str) -> list[str]:
    """
    capture = cv2.VideoCapture(vid_path)
    FPS = int(capture.get(cv2.CAP_PROP_FPS))
    INTERVAL = int(FPS / 2)
    predictor = DefaultPredictor(config)

    words = []
    n = 0
    while (True):
        ret, frame = capture.read()

        if not ret:
            return words
        
        if n == INTERVAL:
            frame = cv2.resize(frame, (224, 224))
            outputs = predictor(frame)['instances']._fields['pred_classes'].tolist()
            for out in outputs:
                mapped = mapper(out)
                if len(words) != 0:
                    if mapped != words[-1]:
                        words.append(mapped)
                else:
                    words.append(mapped)
            n = 0
        n += 1
    """
    return ['slowo1', 'slowo2']

@app.route('/', methods=['POST'])
def upload_video():
    # Validation
    if 'file' not in request.files:
        abort(422, description="No file part")

    file = request.files['file']
    if file.filename == '':
        abort(422, description="No file selected for uploading")

    fileExtension = secure_filename(file.filename).split('.')[-1]
    if fileExtension != 'png':
        abort(422, description="Invalid file format")

    # Creating file
    filename = str(uuid.uuid4())
    filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename + '.' + fileExtension)
    file.save(filePath)

    # Running ML
    result = predict('config', filePath)

    # Tidying up
    os.remove(filePath)

    return {
        "preds": result
    }

if __name__ == "__main__":
    app.run()
