"""
Assembles the confirmed Stage 3 selections into the JSON asset manifest
described in the README.
"""

from __future__ import annotations


def build_manifest(
    beta: list[float] | None,
    face_id: str,
    face_label: str,
    hair_id: str,
    hair_label: str,
    hair_color: str,
    upper_id: str,
    upper_label: str,
    upper_fit: str,
    upper_color: str,
    lower_id: str,
    lower_label: str,
    lower_fit: str,
    lower_color: str,
    accessories: list[dict],
) -> dict:
    """
    Returns the canonical asset manifest dict. Garments that the user
    set to 'none' (empty id) are omitted.
    """
    manifest: dict = {
        "schema_version": "1.0",
        "body": {
            "mesh": "smplx_neutral",
            "beta": [round(b, 6) for b in beta] if beta else [],
        },
        "face": {
            "mesh": "flame_base",
            "shape_label": face_label,
            "mesh_variant": face_id,
        },
        "hair": {
            "mesh": hair_id,
            "style_label": hair_label,
            "color_sample": hair_color,
        },
        "garments": [],
        "accessories": [],
    }

    if upper_id and upper_id != "none":
        manifest["garments"].append({
            "mesh": upper_id,
            "label": upper_label,
            "fit": upper_fit,
            "layer": "base",
            "color_sample": upper_color,
        })

    if lower_id and lower_id != "none":
        manifest["garments"].append({
            "mesh": lower_id,
            "label": lower_label,
            "fit": lower_fit,
            "layer": "base",
            "color_sample": lower_color,
        })

    for acc in accessories:
        if acc.get("id") and acc.get("id") != "none":
            manifest["accessories"].append({
                "mesh": acc["id"],
                "label": acc["label"],
                "category": acc["category"],
            })

    return manifest
