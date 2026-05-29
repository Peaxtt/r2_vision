from flask import Blueprint, request, jsonify
from core.pathfinding import (
    generate_random_valid_state,
    validate_full_setup,
    build_forest_graph,
    build_n_pick_route_candidates,
)

mission_bp = Blueprint("mission", __name__)


@mission_bp.route("/api/random", methods=["POST"])
def api_random():
    data = request.json or {}
    n_picks = int(data.get("n_picks", 2))
    allow_diagonal = bool(data.get("allow_diagonal", False))
    strict_clearance = bool(data.get("strict_clearance", False))
    robot_type = str(data.get("robot_type", "R2"))

    # Context-Aware Brute Force
    for _ in range(2000):
        occ, heights = generate_random_valid_state()
        graph = build_forest_graph(heights)
        routes = build_n_pick_route_candidates(
            graph, occ, n_picks, allow_diagonal, strict_clearance, robot_type
        )
        if routes:
            return jsonify({"occupancy": occ, "heights": heights})

    occ, heights = generate_random_valid_state()
    return jsonify({"occupancy": occ, "heights": heights})


@mission_bp.route("/api/validate", methods=["POST"])
def api_validate():
    data = request.json
    if not data:
        return jsonify({"valid": False, "errors": ["No data"]}), 400
    occ = {int(k): str(v) for k, v in data["occupancy"].items()}
    hts = {int(k): int(v) for k, v in data["heights"].items()}
    val, errs = validate_full_setup(occ, hts)
    return jsonify({"valid": val, "errors": errs})


@mission_bp.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json
    if not data:
        return jsonify([]), 400

    occ = {int(k): str(v) for k, v in data["occupancy"].items()}
    hts = {int(k): int(v) for k, v in data["heights"].items()}
    n_picks = int(data.get("n_picks", 2))
    allow_diagonal = bool(data.get("allow_diagonal", False))
    strict_clearance = bool(data.get("strict_clearance", False))
    robot_type = str(data.get("robot_type", "R2"))

    graph = build_forest_graph(hts)
    routes = build_n_pick_route_candidates(
        graph, occ, n_picks, allow_diagonal, strict_clearance, robot_type
    )

    res = [
        {
            "name": r.name,
            "picked_targets": r.picked_targets,
            "actions": r.actions,
            "score": r.score,
            "final_block": r.final_block,
            "exit_block": r.exit_block,
        }
        for r in routes[:3]
    ]
    return jsonify(res)
