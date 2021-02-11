from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename
from forms import DemoModel
from detectron2.engine import DefaultPredictor
from model import get_model, model_predict
from helpers import generate_filename
import os



__author__ = 'Ivan Timoshenko'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#Get model
predictor, metadata = get_model()

# FLASK START#######################
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asfesf45p34g#UOoi3op8dsgfdsgw'
app.config['UPLOAD_FOLDER'] = 'static/upload'
app.config['MAX_CONTENT_PATH'] = 15000000
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        file_name = f.filename
        extension = os.path.splitext(file_name)[1]
        file_name = secure_filename(generate_filename(extension))
        file_fullname = os.path.join(APP_ROOT,
                                     app.config['UPLOAD_FOLDER'],
                                     file_name)
        f.save(file_fullname)

        out_img_path = model_predict(predictor, metadata, file_fullname, extension)
        return render_template('index.html',
                               form=None,
                               out_img=os.path.join(os.path.basename(app.config['UPLOAD_FOLDER']),
                                                    os.path.basename(out_img_path)))

    form = DemoModel()
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)