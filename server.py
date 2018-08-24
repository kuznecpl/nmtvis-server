from flask import Flask, jsonify, request, send_file, redirect, url_for
import os
import pickle
import json
import random
from uuid import uuid4
from flask_cors import CORS
import subprocess
import torch
import seaborn as sns
import matplotlib
from werkzeug.utils import secure_filename
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity
)
from flask_sqlalchemy import SQLAlchemy

matplotlib.use('Agg')
import matplotlib.pyplot as plt

UPLOAD_FOLDER = './uploads'
DOCUMENTS_FOLDER = './documents'
ALLOWED_EXTENSIONS = set(['txt'])
DB_NAME = 'seq2seq.db'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JWT_SECRET_KEY'] = "supersecretkeyhastochange"
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/science/' + DB_NAME
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
jwt = JWTManager(app)

from shared import db
from db_models import User, Document as DBDocument

with app.app_context():
    db.init_app(app)

    import os.path

    if not os.path.exists(DB_NAME):
        db.create_all()
        admin = User(username="paul", password="sanja")
        db.session.add(admin)
        db.session.commit()

from myseq2seq.seq2seq import seq2seq_model
from myseq2seq.document import Document as Document, Sentence
from myseq2seq.scorer import Scorer
from myseq2seq.keyphrase_extractor import DomainSpecificExtractor


@app.route('/api/auth/login', methods=['POST'])
def login():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if not username:
        return jsonify({"msg": "Missing username parameter"}), 400
    if not password:
        return jsonify({"msg": "Missing password parameter"}), 400

    maybe_user = User.query.filter_by(username=username).first()
    if not maybe_user or maybe_user.password != password:
        return jsonify({"msg": "Bad username or password"}), 401

    # Identity can be any data that is json serializable
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token, status="success", username=username), 200


@app.route('/api/auth/register', methods=['POST'])
def register():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if not username:
        return jsonify({"msg": "Missing username parameter"}), 400
    if not password:
        return jsonify({"msg": "Missing password parameter"}), 400

    maybe_user = User.query.filter_by(username=username).first()
    if maybe_user:
        return jsonify({"msg": "Username already registered"}), 409
    new_user = User(username=username, password=password)

    # Create sample document
    if os.path.isfile(DOCUMENTS_FOLDER + "/document-SAMPLE.document"):
        sample_document = get_document("SAMPLE")
        id = uuid4()
        dbDocument = DBDocument(id=id, name="Sample", user=new_user)
        save_document(sample_document, id)
        db.session.add(dbDocument)

    db.session.add(new_user)
    db.session.commit()

    # Identity can be any data that is json serializable
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token, status="success", username=username), 200


@app.route('/protected', methods=['GET'])
@jwt_required
def protected():
    # Access the identity of the current user with get_jwt_identity
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200


def addTranslation(root, translation):
    if not translation.words:
        return

    for child in root["children"]:
        if child["name"] == translation.words[0]:
            addTranslation(child, translation.slice())
            return

    node = {"name": translation.words[0], "logprob": translation.log_probs[0], "children": [],
            "attn": translation.attns[0], "candidates": translation.candidates[0], "is_golden": translation.is_golden,
            "is_unk": translation.is_unk[0]}
    root["children"].append(node)
    addTranslation(node, translation.slice())


def translationsToTree(translations):
    root = {"name": "root", "logprob": 0, "children": [], "candidates": [], "is_golden": False}

    for translation in translations:
        addTranslation(root, translation)

    if root["children"]:
        return root["children"][0]
    else:
        return root


@app.route("/opennmt", methods=['GET', 'POST'])
@jwt_required
def hello():
    data = request.get_json()
    sentence = data["sentence"]

    # Write sentence to source file
    with open("OpenNMT-py-base/source.txt", "w") as f:
        f.write(sentence)

    # Call OpenNMT model to translate source file
    subprocess.run(["sh", "OpenNMT-py-base/translate.sh"])

    # Read translation from prediction file
    translations = ""
    with open("OpenNMT-py-base/pred.txt", "r") as f:
        translations = f.read().replace("&apos;", "'").split("\n")[:-1]

    translation = translations[0]
    tokenized_translations = [t.split(" ") + ["<EOS>"] for t in translations]

    beam = translationsToTree(tokenized_translations)

    # Parse attention weights
    attn = torch.load("OpenNMT-py-base/attention.weights")[0].numpy()
    plotResult = translation.replace("&apos;", "'") + " <EOS>"
    heatmap = plt.axes()
    sns.heatmap(attn, xticklabels=sentence.split(" "), yticklabels=plotResult.split(" "), ax=heatmap)
    # heatmap.set_title("Attention Matrix")
    plt.savefig("heatmap.png", bbox_inches='tight')
    plt.clf()

    attn = attn.tolist()
    # img = open("heatmap.png", "rb").read()

    res = {}
    res["sentence"] = sentence
    res["translation"] = translation + " <EOS>"
    res["attention"] = attn
    res["beam"] = beam

    return jsonify(res)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/correctTranslation", methods=["POST"])
@jwt_required
def correctTranslation():
    data = request.get_json()
    translation = data["translation"]
    beam = data["beam"]
    document_unk_map = data["document_unk_map"]
    attention = data["attention"]
    document_id = data["document_id"]
    sentence_id = data["sentence_id"]

    document = get_document(document_id)

    extractor = DomainSpecificExtractor(source_file=document.filepath,
                                        train_source_file="myseq2seq/data/wmt14/train.tok.clean.bpe.32000.de",
                                        train_vocab_file="myseq2seq/train_vocab.pkl")
    keyphrases = extractor.extract_keyphrases()

    import uuid

    for key in document_unk_map:
        if key not in document.unk_map:
            document.unk_map[key] = document_unk_map[key]
        else:
            # Merge list values
            document.unk_map[key] = list(set(document.unk_map[key]) | set(document_unk_map[key]))

    sentence = document.sentences[int(sentence_id)]
    sentence.translation = translation
    sentence.corrected = True
    sentence.attention = attention
    sentence.beam = beam

    scorer = Scorer()
    score = scorer.compute_scores(sentence.source, sentence.translation, attention, keyphrases)
    score["order_id"] = sentence.score["order_id"]
    sentence.score = score

    document.sentences[int(sentence_id)] = sentence

    save_document(document, document_id)

    from myseq2seq.train import train_iters
    pairs = [sentence.source, sentence.translation[:-4]]
    print(pairs)
    # train_iters(seq2seq_model.encoder, seq2seq_model.decoder, seq2seq_model.input_lang, seq2seq_model.output_lang,
    #           pairs, batch_size=1, print_every=1, n_epochs=1)

    return jsonify({})


@app.route("/api/documents/<document_id>/sentences", methods=["GET"])
@jwt_required
def getSentences(document_id):
    document = get_document(document_id)

    sentences = []

    for sentence in document.sentences:
        sentences.append(
            {"id": str(sentence.id), "source": sentence.source, "translation": sentence.translation,
             "beam": sentence.beam,
             "score": sentence.score,
             "attention": sentence.attention,
             "corrected": sentence.corrected})

    return jsonify(sentences)


@app.route("/api/documents", methods=["GET"])
@jwt_required
def getDocuments():
    import pickle

    res = []

    user = User.query.filter_by(username=get_jwt_identity()).first()

    if not user:
        return jsonify([]), 401

    for db_document in user.documents:
        document = get_document(db_document.id)
        document_map = {"id": db_document.id, "name": db_document.name, "keyphrases": document.keyphrases}
        res.append(document_map)

    return jsonify(res)


def save_document(document, document_id):
    pickle.dump(document, open(DOCUMENTS_FOLDER + "/document-" + str(document_id) + ".document", "wb"))


def get_document(document_id):
    return pickle.load(open(os.path.join(DOCUMENTS_FOLDER, "document-" + str(document_id) + ".document"), "rb"))


@app.route("/api/documents/<document_id>/sentences/<sentence_id>", methods=["GET"])
@jwt_required
def getTranslationData(document_id, sentence_id):
    data = request.get_json()

    document = get_document(document_id)

    sentence = document.sentences[int(sentence_id)]

    translation, attn, beam = sentence.translation, sentence.attention, sentence.beam
    document_map = {"inputSentence": sentence.source, "translation": translation, "attention": attn,
                    "beam": beam, "document_unk_map": document.unk_map}
    return jsonify(document_map)


@app.route("/api/documents/<document_id>/sentences/<sentence_id>/corrected", methods=["POST"])
@jwt_required
def setCorrected(document_id, sentence_id):
    data = request.get_json()
    corrected = data["corrected"]

    document = get_document(document_id)

    sentence = document.sentences[int(sentence_id)]
    sentence.corrected = corrected
    save_document(document, document_id)

    return jsonify({"status": "ok"})


@app.route("/upload", methods=['POST'])
@jwt_required
def documentUpload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        document_name = request.args.get("document_name")
        id = uuid4()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        user = User.query.filter_by(username=get_jwt_identity()).first()
        dbDocument = DBDocument(id=id, name=document_name, user=user)

        document = Document(str(id), document_name, dict(), filepath)
        sentences = document.load_content(filename)

        with open(filepath, "w") as f:
            for i, sentence in enumerate(sentences):
                f.write(sentence.replace("@@ ", "") + "\n" if i < len(sentences) - 1 else "")

        extractor = DomainSpecificExtractor(source_file=filepath,
                                            train_source_file="myseq2seq/data/wmt14/train.tok.clean.bpe.32000.de",
                                            train_vocab_file="myseq2seq/train_vocab.pkl")
        keyphrases = extractor.extract_keyphrases()

        scorer = Scorer()

        for i, source in enumerate(sentences):
            translation, attn, translations = seq2seq_model.translate(source)

            beam = translationsToTree(translations)

            score = scorer.compute_scores(source, " ".join(translation), attn, keyphrases)
            score["order_id"] = i
            sentence = Sentence(i, source, " ".join(translation), attn, beam, score)

            document.sentences.append(sentence)

        keyphrases = [{"name": k, "occurrences": f, "active": False} for (k, f) in keyphrases]
        document.keyphrases = keyphrases
        db.session.add(dbDocument)
        db.session.commit()

        save_document(document, id)

        return jsonify({})
    return jsonify({})


@app.route("/beamUpdate", methods=['POST'])
@jwt_required
def beamUpdate():
    data = request.get_json()
    sentence = data["sentence"]
    beam_size = int(data["beam_size"])
    beam_length = float(data["beam_length"])
    beam_coverage = float(data["beam_coverage"])
    attentionOverrideMap = data["attentionOverrideMap"]
    correctionMap = data["correctionMap"]
    unk_map = data["unk_map"]

    translation, attn, translations = seq2seq_model.translate(sentence, beam_size,
                                                              beam_length=beam_length,
                                                              beam_coverage=beam_coverage,
                                                              attention_override_map=attentionOverrideMap,
                                                              correction_map=correctionMap, unk_map=unk_map)
    beam = translationsToTree(translations)
    res = {}
    res["beam"] = beam

    return jsonify(res)


@app.route("/attentionUpdate", methods=['POST'])
@jwt_required
def attentionUpdate():
    data = request.get_json()
    sentence = data["sentence"]
    attentionOverrideMap = data["attentionOverrideMap"]
    correctionMap = data["correctionMap"]
    unk_map = data["unk_map"]
    beam_size = int(data["beam_size"])
    beam_length = float(data["beam_length"])
    beam_coverage = float(data["beam_coverage"])

    translation, attn, translations = seq2seq_model.translate(sentence, beam_size,
                                                              beam_length=beam_length,
                                                              beam_coverage=beam_coverage,
                                                              attention_override_map=attentionOverrideMap,
                                                              correction_map=correctionMap, unk_map=unk_map)
    beam = translationsToTree(translations)
    res = {}
    res["beam"] = beam

    return jsonify(res)


@app.route("/wordUpdate", methods=['POST'])
@jwt_required
def wordUpdate():
    data = request.get_json()
    sentence = data["sentence"]
    attentionOverrideMap = data["attentionOverrideMap"]
    correctionMap = data["correctionMap"]
    unk_map = data["unk_map"]
    beam_size = int(data["beam_size"])
    beam_length = float(data["beam_length"])
    beam_coverage = float(data["beam_coverage"])

    translation, attn, translations = seq2seq_model.translate(sentence, beam_size,
                                                              beam_length=beam_length,
                                                              beam_coverage=beam_coverage,
                                                              attention_override_map=attentionOverrideMap,
                                                              correction_map=correctionMap, unk_map=unk_map)
    beam = translationsToTree(translations)
    res = {}
    res["beam"] = beam

    return jsonify(res)


@app.route("/", methods=['GET', 'POST'])
@jwt_required
def translate():
    data = request.get_json()
    sentence = data["sentence"]
    beam_size = int(data["beam_size"])
    beam_length = float(data["beam_length"])
    beam_coverage = float(data["beam_coverage"])

    translation, attn, translations = seq2seq_model.translate(sentence, beam_size, beam_length=beam_length,
                                                              beam_coverage=beam_coverage, apply_bpe=False)

    res = {}
    res["sentence"] = sentence
    res["translation"] = " ".join(translation)
    res["attention"] = attn

    beam = translationsToTree(translations)
    res["beam"] = beam

    return jsonify(res)


@app.route("/api/documents/<document_id>/retrain", methods=['POST'])
@jwt_required
def retrain(document_id):
    document = get_document(document_id)

    pairs = []
    for sentence in document.sentences:
        # Remove EOS at end
        if sentence.corrected:
            pairs.append([sentence.source, sentence.translation[:-4]])

    if len(pairs) < 2:
        return jsonify({})

    from myseq2seq.train import retrain_iters
    retrain_iters(seq2seq_model, pairs, [], batch_size=min(256, len(pairs)), print_every=1, n_epochs=20,
                  learning_rate=0.00001)

    return jsonify({})


@app.route("/api/documents/<document_id>/translate", methods=['POST'])
@jwt_required
def retranslate(document_id):
    document = get_document(document_id)
    scorer = Scorer()
    extractor = DomainSpecificExtractor(source_file=document.filepath,
                                        train_source_file="myseq2seq/data/wmt14/train.tok.clean.bpe.32000.de",
                                        train_vocab_file="myseq2seq/train_vocab.pkl")
    keyphrases = extractor.extract_keyphrases()

    for i, sentence in enumerate(document.sentences):
        if sentence.corrected:
            continue

        translation, attn, translations = seq2seq_model.translate(sentence.source)

        beam = translationsToTree(translations)

        score = scorer.compute_scores(sentence.source, " ".join(translation), attn, keyphrases)
        score["order_id"] = i

        sentence.translation = " ".join(translation)
        sentence.beam = beam
        sentence.score = score
        sentence.attention = attn

    save_document(document, document_id)
    return jsonify({})


@app.route("/api/experiments/next", methods=['POST'])
@jwt_required
def nextExperimentSentence():
    user = User.query.filter_by(username=get_jwt_identity()).first()
    data = request.get_json()
    experiment_metrics = data["experimentMetrics"]

    if not user:
        return jsonify({}), 401

    dbDocument = DBDocument.query.filter_by(user=user, name="Sample").first()
    document = get_document(dbDocument.id)
    current_sentence = document.sentences[user.current_experiment_index]
    current_sentence.experiment_metrics = experiment_metrics
    save_document(document, dbDocument.id)

    user.current_experiment_index += 1

    db.session.add(user)
    db.session.commit()

    if user.current_experiment_index >= len(document.sentences):
        return jsonify(status="finished")

    next_sentence = document.sentences[user.current_experiment_index]
    next_index = next_sentence.id
    experiment_type = next_sentence.experiment_type

    return jsonify(status="in_progress", documentId=dbDocument.id, sentenceId=next_index,
                   experimentType=experiment_type)


@app.route("/api/experiments/surveydata", methods=['POST'])
@jwt_required
def sendSurveyData():
    user = User.query.filter_by(username=get_jwt_identity()).first()
    data = request.get_json()

    if not user:
        return jsonify({}), 401

    user.surveydata = json.dumps(data)
    db.session.add(user)
    db.session.commit()

    return jsonify()


@app.route("/api/experiments/experimentdata", methods=['GET'])
@jwt_required
def getExperimentData():
    user = User.query.filter_by(username=get_jwt_identity()).first()
    data = request.get_json()

    if not user:
        return jsonify({}), 401

    dbDocument = DBDocument.query.filter_by(user=user, name="Sample").first()
    document = get_document(dbDocument.id)

    result = []

    for sentence in document.sentences:
        result.append(sentence.experiment_metrics)

    return jsonify({"metrics": result, "survey": json.loads(user.surveydata)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, use_reloader=False)
