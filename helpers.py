import uuid
import os
from flask import current_app

def generate_filename(extension):
    xx = str(uuid.uuid4()) + extension
    if os.path.isfile(os.path.join(current_app.config['UPLOAD_FOLDER'], xx)):
        return generate_filename(extension)
    return xx