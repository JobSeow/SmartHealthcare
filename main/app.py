from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import pickle,os,cv2,datetime,glob2
from tensorflow.keras.models import load_model
import numpy as np
import prediction as training_script

########## Flask Configurations #########
app = Flask(__name__)
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}
CATEGORIES = ["no_problem", "problem"]
app.secret_key = 'SECRET KEY'


######## Needed functions ###############
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare(image):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(image)  # read in the image, convert to grayscale
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # cv2 uses BGR rather than RGB, need to convert
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # return the image with shaping that TF wants.


######## Routes ###############
@app.route("/")
def hello():
    return 'Hello, Flask!'

@app.route('/train', methods=['POST'])
def train():
    print("Training Started.........")
    training_script.train()
    return "Finished training!"

@app.route('/images', methods=['POST']) 
def start_images(): 

    print('images hit')
    return "images hit"

@app.route('/predict', methods=['POST'])
def predict():
    print("predict hit!")
    print("loading model......")
    list_of_models = glob2.glob("./models/*")
    model_to_load=list_of_models[-1]
    model = load_model(os.path.join("./",model_to_load))
    if 'file' not in request.files:
        flash('No file part')
        return 'No file part'
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join('./test','test.jpg'))
        print(filename + " loaded. Now Predicting.......")

    
    prediction = model.predict([prepare('./test/test.jpg')]) 
    print("Prediction is "+ CATEGORIES[int(prediction[0][0])])
    return "Prediction is "+ CATEGORIES[int(prediction[0][0])]

@app.route('/upload', methods=['POST'])
def upload_file():
    add_to_folder = ''
    print(request.files)
    if 'no_problem' in request.files:
        file = request.files['no_problem']
        add_to_folder = 'no_problem/'
    elif 'problem' in request.files:
        file = request.files['problem']
        add_to_folder = 'problem/'
    else:
        flash('No file part')
        return 'No file part'


    if file.filename == '':
        flash('No selected file')
        
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = add_to_folder+ filename
        file.save(os.path.join('./training_files',filename))
        print(" added to" +filename  )
    print("somthing")
    return " added to " +filename 


if __name__ == '__main__':
    app.run(debug=True)