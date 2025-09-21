import os
from flask import Flask, render_template, send_from_directory, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from deepface import DeepFace

# --- Flask setup ---
app = Flask(__name__)
app.config["SECRET_KEY"] = "randomstring"

# Folder to save uploaded photos
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Reference folder with known images
REFERENCE_DIR = "./static/bills_db"
os.makedirs(REFERENCE_DIR, exist_ok=True)

# --- Form definition ---
class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(["jpg", "jpeg", "png"], "Only images are allowed"),
            FileRequired("File field should not be empty"),
        ]
    )
    submit = SubmitField("Upload")


# --- Route to serve uploaded files ---
@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# --- Main route ---
@app.route("/", methods=["GET", "POST"])
def uploadimage():
    form = UploadForm()
    file_url = None
    similarity_results = []

    if form.validate_on_submit():
        # Save uploaded file
        file = form.photo.data
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        file_url = url_for("get_file", filename=filename)

        # Collect valid reference images (only those with faces)
        reference_images = [
            os.path.join(REFERENCE_DIR, f)
            for f in os.listdir(REFERENCE_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        valid_images = []
        for img_path in reference_images:
            try:
                _ = DeepFace.represent(img_path=img_path, model_name="VGG-Face")
                valid_images.append(img_path)
            except Exception:
                print(f"Skipping {img_path}, no face detected.")

        # Compare uploaded image against valid reference images
        results = []
        for ref_img in valid_images:
            try:
                result = DeepFace.verify(
                    img1_path=save_path,
                    img2_path=ref_img,
                    model_name="VGG-Face",
                )
                similarity = 1 - result["distance"]  # similarity score
                results.append((os.path.basename(ref_img), similarity))
            except Exception as e:
                print(f"Error comparing {ref_img}: {e}")

        # Sort results by similarity
        similarity_results = sorted(results, key=lambda x: x[1], reverse=True)

    return render_template(
        "index.html",
        form=form,
        file_url=file_url,
        similarity_results=similarity_results,
    )


# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)
