import uuid
import os
from flask import current_app
import time



def generate_filename(extension):
    xx = str(uuid.uuid4()) + extension
    if os.path.isfile(os.path.join(current_app.config['UPLOAD_FOLDER'], xx)):
        return generate_filename(extension)
    return xx

def remove_old_files(dir, min):
    now_time = time.time()
    age_time = now_time - 60 * min

    for path, dirs, files in os.walk(dir):
        for file in files:
            file_name = os.path.join(path, file)
            file_create_time = os.path.getctime(file_name)
            if file_create_time < age_time:
                os.remove(file_name)