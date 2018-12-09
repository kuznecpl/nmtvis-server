from shared import db
from sqlalchemy_utils import PasswordType, UUIDType
from sqlalchemy_utils import force_auto_coercion

force_auto_coercion()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(
        PasswordType(schemes=['pbkdf2_sha512', ])
    )
    documents = db.relationship('Document', backref="user", lazy=True)
    surveydata = db.Column(db.String(5000), nullable=True)
    current_experiment_index = db.Column(db.Integer, default=0)

    def __repr__(self):
        return '<User %r>' % self.username


class Document(db.Model):
    id = db.Column(UUIDType(binary=False), primary_key=True)
    name = db.Column(db.Unicode(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    @property
    def path(self):
        return "document-" + self.id + ".document"

    def __repr__(self):
        return '<Document %r>' % self.name


'''
class Sentence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.Unicode(1000), nullable=False)
    translation = db.Column(db.Unicode(1000), nullable=False)
    document_id = db.Column(UUIDType(binary=False), db.ForeignKey('document.id'), nullable=False)
    corrected = db.Column(db.Boolean, unique=False, default=False)

    def __repr__(self):
        return '<Sentence %r>' % self.source
'''
