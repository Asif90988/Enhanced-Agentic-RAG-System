from flask import Flask
from routes.query_routes import query_routes

app = Flask(__name__)
app.register_blueprint(query_routes, url_prefix="/api")  # only ONCE

if __name__ == '__main__':
    app.run(port=5010, debug=True)
