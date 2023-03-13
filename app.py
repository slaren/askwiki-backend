import logging

import flask
from flask_cors import CORS

import gptwiki
import tindex

app = flask.Flask(__name__)
CORS(app)  # Allow CORS for all routes


@app.route("/suggest")
def suggest():
    query = flask.request.args.get("q")
    if not query:
        return []
    results = tindex.suggest_titles(query, 5)
    return results


@app.route("/search")
def search():
    query = flask.request.args.get("q")
    if not query:
        return []
    results = tindex.search_titles(query)
    return results


@app.route("/answer")
def answer():
    query = flask.request.args.get("q")
    article = flask.request.args.get("a")
    if not query or not article:
        flask.abort(400)
    gen = gptwiki.get_answer(article, query)
    answer = "".join(gen)
    return flask.jsonify(answer)


def event_stream(gen):
    for data in gen:
        yield "event: fragment\n"
        yield f"data: {flask.json.dumps(data)}\n\n"
    yield "event: done\n"
    yield "data: {}\n\n"


@app.route("/stream-answer")
def stream_answer():
    query = flask.request.args.get("q")
    article = flask.request.args.get("a")
    if not query or not article:
        flask.abort(400)

    gen = gptwiki.get_answer(article, query)
    response = flask.Response(
        event_stream(gen),
        mimetype="text/event-stream",
    )
    return response


if __name__ == "__main__":
    for logger_name in ("server", "gptwiki", "tindex"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

    logging.info("Starting debug server")

    from dotenv import load_dotenv

    load_dotenv("secrets.env")

    app.run(debug=True)
