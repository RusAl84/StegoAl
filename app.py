from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin
import os
import time
import urllib.request
import processing_stego
from flask import send_from_directory
import config
a_path=config.a_path

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def dafault_route():
    return 'API'


@app.route('/uploadimg', methods=['POST'])
@cross_origin()
def uploadimg():
    for fname in request.files:
        f = request.files.get(fname)
        print(f)
        # milliseconds = int(time.time() * 1000)
        # filename = str(milliseconds)
        in_filename = a_path + f"uploads\\in.png"
        f.save(in_filename)
        in_param=f"Загружено изображение размером {os.path.getsize(in_filename)} байт"
        out_data={}
        out_data['in_img_path']="http://localhost:5000/uploads/in.png"
        out_data['in_param']=in_param
    return out_data

@app.route('/uploadkey', methods=['POST'])
@cross_origin()
def uploadkey():
    for fname in request.files:
        f = request.files.get(fname)
        print(f)
        # milliseconds = int(time.time() * 1000)
        # filename = str(milliseconds)
        in_filename = a_path + f"uploads\\key1.key"
        f.save(in_filename)


@app.route('/uploads/<path:path>')
def send_photo(path):
    return send_from_directory('uploads', path)

@app.route("/stego_proc", methods=['POST'])
def stego_proc():
    pass
    msg = request.json
    # print(msg)
    text = msg['text']
    img_fileName = a_path + "uploads\\in.png"
    out_filename = a_path + "uploads\\out.png" 
    print(text)
    print(out_filename)
    processing_stego.encode(img_fileName, out_filename, text,4)
    out_param=f"Получено изображение размером {os.path.getsize(out_filename)} байт"
    out_data={}
    out_data['out_img_path'] = "http://localhost:5000/uploads/out.png"
    out_data['out_param'] = out_param
    return out_data


@app.route("/stego_reseach", methods=['POST'])
def stego_reseach():
    pass
    img_fileName = a_path + "uploads\\in.png"
    out_filename = a_path + "uploads\\out.png" 

    print(out_filename)
    processing_stego.stego_reseach()
    out_param=f"Получено изображение размером {os.path.getsize(out_filename)} байт"
    out_data={}
    out_data['ast'] = "http://localhost:5000/uploads/ast.png"
    out_data['ars'] = "http://localhost:5000/uploads/ars.png"
    out_data['asz'] = "http://localhost:5000/uploads/asz.png"
    out_data['amsz'] = "http://localhost:5000/uploads/amsz.png"
    out_data['adif'] = "http://localhost:5000/uploads/adif.png"
    out_data['acomp'] = "http://localhost:5000/uploads/acomp.png"
    return out_data

@app.route("/stego_decode", methods=['POST'])
def stego_decode():
    pass
    img_fileName = a_path + "uploads\\in.png"
    out_filename = a_path + "uploads\\out.png" 
    print(out_filename)
    str1 = processing_stego.decode()
    out_data={}
    out_data['decode'] = str1
    return out_data


@app.route("/clear_db", methods=['GET'])
def clear_db():
    return "ok clear_db"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
# app.run(host="0.0.0.0")