from flask import Flask, render_template, send_from_directory
from core.logger import setup_logger
from api.control import control_bp
from api.mission import mission_bp
from api.telemetry import telemetry_bp
from api.diagnostics import diagnostics_bp

log = setup_logger()


def create_app():
    app = Flask(__name__)
    app.register_blueprint(control_bp)
    app.register_blueprint(mission_bp)
    app.register_blueprint(telemetry_bp)
    app.register_blueprint(diagnostics_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/manifest.json")
    def manifest():
        return send_from_directory("templates", "manifest.json",
                                   mimetype="application/manifest+json")

    return app


if __name__ == "__main__":
    log.info("=== ABU Robocon 2026 Vision UI starting ===")
    log.info(f"Log file: /tmp/abu_vision.log")
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
