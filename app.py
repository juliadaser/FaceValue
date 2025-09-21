import os
from flask import Flask, render_template, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from deepface import DeepFace
import base64

# __Flask setup__
app = Flask(__name__)
app.config["SECRET_KEY"] = "randomstring"

# Reference folder with money bills
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


# --- Main route ---
@app.route("/", methods=["GET", "POST"])
def uploadimage():
    form = UploadForm()
    file_url = None
    similarity_results = []

    if form.validate_on_submit():
        # Read user's selfie into memory
        file = form.photo.data
        img_bytes = file.read()

        # Convert selfie base64 to display on webpage
        file_url = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")

        # Save selfie temporarily to disk just for DeepFace
        temp_filename = f"temp_{secure_filename(file.filename)}"
        with open(temp_filename, "wb") as f:
            f.write(img_bytes)

        # Collect money bills (only those where faces can be detected)
        reference_images = [
            os.path.join(REFERENCE_DIR, f)
            for f in os.listdir(REFERENCE_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        valid_images = [] #valid money bills
        for img_path in reference_images:
            try:
                _ = DeepFace.represent(img_path=img_path, model_name="VGG-Face")
                valid_images.append(img_path)
            except Exception:
                print(f"Skipping {img_path}, no face detected.")

        # Compare selfie against list of valid money bills
        results = []
        for ref_img in valid_images:
            try:
                result = DeepFace.verify(
                    img1_path=temp_filename,
                    img2_path=ref_img,
                    model_name="VGG-Face",
                )
                similarity = 1 - result["distance"]  # similarity score
                results.append((os.path.basename(ref_img), similarity))
            except Exception as e:
                print(f"Error comparing {ref_img}: {e}")

        # Delete user's selfie
        os.remove(temp_filename)

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
