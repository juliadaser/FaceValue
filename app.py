import re
import os
from datetime import datetime
from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from deepface import DeepFace

app = Flask(__name__)
app.config["SECRET_KEY"] = 'randomstring'
app.config["UPLOADED_PHOTOS_DEST"] = 'uploads' #name of created folder

photos = UploadSet('photos', IMAGES) # pictures the user uploads
configure_uploads(app, photos)

REFERENCE_DIR = "./static/bills_db" # folder of bills
reference_images = [
    os.path.join(REFERENCE_DIR, f)
    for f in os.listdir(REFERENCE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

valid_images = [] # for some reason, a face is not detected on all bills!! Therefore, create a new list of bills that are recognized by deepface
for img_path in reference_images:
    try:
        _ = DeepFace.represent(img_path=img_path, model_name="VGG-Face")
        valid_images.append(img_path)
    except Exception as e:
        print(f"Skipping {img_path}, no face detected.")

class UploadForm(FlaskForm):
    photo = FileField(
        validators = {
            FileAllowed(photos, "Only images are allowed"),
            FileRequired('File Field should not be empty')

        }
    )
    submit = SubmitField('upload')


@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)



@app.route("/", methods=['GET', 'POST'])
def uploadimage():
    form = UploadForm()
    similarity = None
    if form.validate_on_submit(): # if photo is submitted and contains no errors
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)

        # Path to uploaded file
        uploaded_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], filename)
        # Path to reference image
        reference_path = "static/reference.png"

        # Run DeepFace verification
        result = DeepFace.verify(
            img1_path=uploaded_path,
            img2_path=reference_path,
            model_name="VGG-Face",  # default, but you can try "Facenet", "ArcFace", etc.
        )

        similarity = 1 - result["distance"]

        print("Verification result:", result)
        print("Similarity score:", similarity)

    else:
        file_url = None
    return render_template("index.html", form=form, file_url=file_url, similarity=similarity) # sending info back to frontend

if __name__ == '__main__':
    app.run(debug=True)