from flask import Flask,flash,request,redirect,render_template
from werkzeug.utils import secure_filename
from train import Training
import os
app = Flask(__name__)
trainObj = Training()

UPLOAD_FOLDER = 'static/images/'
ALLOWED_EXT = ['png','jpg','jpeg']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/',methods=['GET'])
def home():
    return render_template('indexx.html')

@app.route('/upload',methods=['POST'])
def upload():
    trainObj.deleteFiles()
    if request.method == 'POST':
        
        print(request)
        if 'image' not in request.files:
            return 'File not present in request url'
            redirect(request.url)
        file = request.files['image']
        if file.filename =='':
            return 'File not selected'
            redirect(request.url)
        fileExt = file.filename.split('.')[1]
        if file and fileExt in ALLOWED_EXT:
            filename = secure_filename(file.filename)
            filePath = os.path.join(UPLOAD_FOLDER,filename)
            file.save(filePath)
            fruitClass = trainObj.predict(filePath)
            fruitName = fruitClass[0]
            fruitLife = 0
            if len(fruitClass) == 2:
                fruitLife = fruitClass[1]
            return render_template('indexx.html',src=filePath,fruitName=fruitName,fruitLife=fruitLife)


if __name__ == '__main__':
    app.run()

