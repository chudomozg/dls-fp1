from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired

class DemoModel(FlaskForm):
    file = FileField(label='select image',
                     description='Image for testing Segmentation Model',
                     validators=[InputRequired()])
    submit = SubmitField(label='Show me')
