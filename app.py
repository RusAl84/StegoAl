from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin
import os
import time
import urllib.request
import processing_stego
from flask import send_from_directory

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
        full_filename = f"./uploads/in.png"
        f.save(full_filename)
    return "http://localhost:5000/uploads/in.png"

@app.route('/uploads/<path:path>')
def send_photo(path):
    return send_from_directory('uploads', path)

@app.route("/stego_proc", methods=['GET'])
def stego_proc():
    pass
    msg = request.json
    # print(msg)
    text = msg['text']
    img_fileName = "http://localhost:5000/uploads/in.png"
    out_filename = "http://localhost:5000/uploads/out.png"
    processing_stego.encode(img_fileName, out_filename, text,4)
    return "http://localhost:5000/uploads/key.dat"


@app.route("/get_pattern_add", methods=['POST'])
def get_pattern_add():
    pass
    # msg = request.json
    # print(msg)
    # data = process_nlp.add_data(msg['text'])
    # data = process_nlp.add_print_text(data)
    # print()
    # print(data)
    # return data


@app.route('/findae', methods=['POST'])
def findae():
    pass
    #     if request.method == 'POST':
    # msg = request.json
    # print(msg)
    # filename = msg['filename']
    # ttype=msg['type']
    # # filename="d:/ml/chat/andromedica1.json"
    # save_filename="./data_proc.json"
    # # data_proc(filename, save_filename, 32)
    # # find_cl(save_filename)
    # data = process_nlp.find_type("./find_data.json", ttype)
    # print(data)
    # return data


@app.route("/clear_db", methods=['GET'])
def clear_db():
    return "ok clear_db"



if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
# app.run(host="0.0.0.0")