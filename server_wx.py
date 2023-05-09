import os
import cv2
import time
import yaml
import uuid
import shutil
import json
from os import listdir
import sqlite3
import datetime
from datetime import timedelta
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from detector import Detector

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(hours=1)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
with open('config.yaml', 'r', encoding='utf-8')as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

@app.route('/upload')
def upload_file():
    return render_template('Index.html')

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in cfg['ALLOWED_EXTENSIONS']

@app.route("/uploader", methods=["GET", "POST"])
def upload():
    file = request.files['file']
    # print(file.filename)
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1]
        # 生成随机文件名
        random_name = '{}.{}'.format(uuid.uuid4().hex, ext)
        # print(random_name)
        savepath = os.path.join(cfg['CACHE_FOLDER'], secure_filename(random_name))
        result_path = os.path.join(cfg['RESULTS_FOLDER'], secure_filename(random_name))
        file.save(savepath)
        shutil.copy(savepath, os.path.join('static', savepath))
        # time-1
        t1 = time.time()
        img = cv2.imread(savepath)
        status, img_result, img_info = detector.detect(img)
        # time-2
        t2 = time.time()
        
        if status == 1:
            cv2.imwrite(result_path, img_result)
            shutil.copy(result_path, os.path.join('static', result_path))
            
        jsonData = {
            'code': status,
            'img_url': os.path.join('static', savepath),
            'result_url': os.path.join('static', result_path),
            'result_json': os.path.join('static', 'results_json\\'+str(secure_filename(random_name).split('.')[0])+'.json'),
            'img_info': img_info,
            'time': '{:.4f}s'.format(t2-t1)
        }
        if not os.path.exists('./results_json'):
            os.mkdir('./results_json')
        jsObj = json.dumps(jsonData, indent=4)
        fileObject = open('./results_json/'+str(secure_filename(random_name).split('.')[0])+'.json', 'w')
        fileObject.write(jsObj)
        fileObject.close()
        shutil.copy('./results_json/'+str(secure_filename(random_name).split('.')[0])+'.json', os.path.join('static', './results_json/'+str(secure_filename(random_name).split('.')[0])+'.json'))
        
        data = {
            "code":1,
            "msg":"success",
            "newslist":[
            {
                "output_url":"http://127.0.0.1:8090/static/results/" + str(random_name)
                },
            {
                "result_arg":str(img_info)
                },
            {
                'detect_time':'{:.4f}s'.format(t2-t1)
                }
            ]
            }
        return json.dumps(eval(str(data)))
    return {
            "code":0,
            "msg":"Method Not Allowed"
                }


if __name__ == '__main__':
    # 热更新组件
    file_path = cfg['DIR_PATH']
    version = str(listdir(file_path)[-1])
    model_name = '/best.pt'
    weights_path = cfg['DIR_PATH'] + version + model_name
    print('============================================')
    print('Model Version：{}，Use Model Weight：{}'.format(version,model_name))
    print('============================================')
    
    detector = Detector(img_size=cfg['IMG_SIZE'], threshold=cfg['THRESHOLD'], weights=weights_path, mydevice=cfg['MYDEVICE'])

    for folder in cfg['FOLDER']:
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join('static', folder), exist_ok=True)

    app.run(host=cfg['HOST'], port=cfg['PORT'], debug=True, threaded=True, processes=1)