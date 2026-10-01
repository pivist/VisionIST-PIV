"""The box definitions are the contract: they must load, validate, and stay
consistent.  A typo here must fail loudly, not ship."""

import pathlib

import pytest

from webui.core import BoxDef, load_registry, RegistryError

BOXES_DIR = pathlib.Path(__file__).resolve().parents[1] / "boxes"


@pytest.fixture(scope="module")
def reg():
    return load_registry(BOXES_DIR)


EXPECTED_IDS = {"clip", "tapnext", "lang_sam", "sbert", "vggt", "moge", "yolo",
                "unimatch", "features"}


def test_out_of_scope_boxes_are_absent(reg):
    """opencv_box predates the shared envelope (Process) contract and is
    intentionally skipped until it migrates — the webui stays contract-only.
    Re-add its definition when the box does."""
    ids = {d.id for d in reg}
    assert "opencv" not in ids


def test_all_defs_load(reg):
    ids = {d.id for d in reg}
    assert ids == EXPECTED_IDS, ids


def test_every_def_has_form_and_results(reg):
    for d in reg:
        assert d.inputs, f"{d.id}: no inputs"
        assert d.results, f"{d.id}: no results"
        assert any(r.field == "*" for r in d.results), f"{d.id}: no '*' fallback visualizer"


def test_flat_boxes_have_no_box_key(reg):
    for d in reg:
        if d.flat_config:
            assert d.box_key is None, d.id
        else:
            assert d.box_key, f"{d.id}: needs a box_key"


def test_clip(reg):
    d = reg.get("clip")
    assert d.box_key == "clip"
    assert d.command.default == "encode"
    assert {f.field for f in d.inputs} == {"images", "texts"}
    assert d.input_mosaic is True   # default: the "inputs" mosaic still shows
    assert any(p.key == "model" for p in d.parameters)


def test_tapnext_is_stateful_sessioned(reg):
    d = reg.get("tapnext")
    assert d.box_key == "tapnext"
    assert d.session is not None and d.session.key == "session_id"
    assert d.session.auto_generate and "reset" in d.session.actions
    assert {a for a in d.command.values} == {"track", "reset", "list"}


def test_lang_sam_section_prompt_is_required(reg):
    d = reg.get("lang_sam")
    sec = {s.key: s for s in d.section}
    assert "text_prompt" in sec and sec["text_prompt"].required
    assert d.box_key == "lang_sam"


def test_vggt_namespaced_and_glb(reg):
    d = reg.get("vggt")
    assert not d.flat_config and d.box_key == "vggt"
    assert set(d.command.values) == {"reconstruct", "reset"}
    assert d.command.default == "reconstruct"
    viz = {r.field: r.visualizer for r in d.results if r.field != "*"}
    assert viz["glb_file"] == "glb"
    assert viz["depth"] == "tensor"


def test_moge_maps_are_visualized_per_item(reg):
    d = reg.get("moge")
    assert not d.flat_config and d.box_key == "moge"
    assert set(d.command.values) == {"infer", "reset"}
    assert d.command.default == "infer"
    params = {p.key for p in d.parameters}
    assert params == {"fov_x", "refine_steps", "resolution_level"}
    # bool param: the wire passes parameters through raw, so a "false"
    # string would read as truthy on the box side -> kept out of the form
    assert "fp16" not in params
    props = {r.params.get("prop") for r in d.results if r.params}
    assert {"depth", "normal", "intrinsics"} <= props
    maps = [r for r in d.results if r.visualizer == "field_map"]
    assert {m.params["prop"] for m in maps} == {"depth", "normal"}
    cloud = [r for r in d.results if r.visualizer == "points"]
    assert cloud
    c = cloud[0]
    # reprojected from the depth map (not a box-provided point array);
    # colors come from the uploaded input image named by `base`
    assert c.params["depth"] == "depth" and c.params["intrinsics"] == "intrinsics"
    assert c.params["mask"] == "mask"
    assert c.base == "images"
    assert any(r.field == "*" for r in d.results)          # fallback renderer


def test_yolo_detection_def(reg):
    d = reg.get("yolo")
    assert d.box_key == "yolo"
    assert set(d.command.values) == {"detect", "reset", "list"}
    assert d.command.default == "detect"
    assert {f.field for f in d.inputs} == {"images", "video"}
    params = {p.key for p in d.parameters}
    # tracking is always on (no batch knob: track mode runs batch=1 per frame)
    assert params == {"weights", "conf", "iou", "imgsz", "classes",
                      "save_annotated", "frame_step", "max_frames"}
    # multi-session like tapnext: per-session tracker state + scoped actions
    assert d.session is not None
    assert d.session.key == "session_id"
    assert d.session.auto_generate is True
    assert set(d.session.actions) == {"reset", "list"}
    viz = {r.field: r.visualizer for r in d.results if r.field != "*"}
    # panel order: annotated video (video input) -> annotated grid (image
    # input only: renders when annotated_video is absent) -> zip download -> table
    assert viz == {
        "annotated_video": "video",
        "annotated": "image_grid",
        "annotated_zip": "download",
        "detections": "table",
    }
    assert [r.field for r in d.results if r.field != "*"][0] == "annotated_video"
    grid = next(r for r in d.results if r.field == "annotated")
    assert grid.visualizer == "image_grid"
    assert grid.only_if_missing == "annotated_video"   # hidden for video input
    zipr = next(r for r in d.results if r.field == "annotated_zip")
    assert zipr.visualizer == "download"
    assert zipr.params.get("filename") == "yolo-annotations.zip"
    tabler = next(r for r in d.results if r.field == "detections")
    assert tabler.params.get("filename") == "yolo-detections.json"
    assert d.input_mosaic is False   # annotated result already shows the input
    assert d.results[-1].field == "*"


def test_unimatch_is_dense_estimation(reg):
    d = reg.get("unimatch")
    assert d.box_key == "unimatch"
    assert d.command.default == "flow"
    assert set(d.command.values) == {"flow", "stereo", "reset"}
    # booleans (pred_bidir_flow / fwd_bwd_check) are deliberately NOT in the
    # form: the wire passes parameters raw, so a select's "false" string would
    # read as truthy on the box side (same rule as moege's fp16)
    params = {p.key for p in d.parameters}
    assert params == {"model", "inference_size", "padding_factor"}
    viz = {r.field: r.visualizer for r in d.results if r.field != "*"}
    assert set(viz.values()) == {"tensor"}          # one generic tensor panel per output
    # spatial semantics are declared data (params.spatial), never inferred:
    # flow quiver over the first input image; scalar maps as heat
    params = {r.field: r.params for r in d.results if r.field != "*"}
    assert params["flow"].get("spatial") == "flow"
    for f in ("occ_fwd", "occ_bwd", "disparity", "depth"):
        assert params[f].get("spatial") == "heat"
    flow = next(r for r in d.results if r.field == "flow")
    assert flow.base == "images"   # quiver over the first uploaded image
    assert d.results[-1].field == "*"


def test_non_process_refused():
    """The contract only allows Process for now — any other method is a
    clear refusal, not a half-working path."""
    from webui.core import BoxDef, CallRequest, CallBuildError, build_call, ArtifactStore
    from webui.core.schema import ActionDef
    d = BoxDef(id="precontract", name="pre", box_key="pre",
               actions=[ActionDef(name="detect", method="DetectSequence")])
    with pytest.raises(CallBuildError, match="not the shared contract RPC"):
        build_call(d, CallRequest(action="detect"), ArtifactStore())


def test_match_is_lenient(reg):
    assert reg.match("tapnext").id == "tapnext"
    assert reg.match("TAPNEXT").id == "tapnext"
    assert reg.match("LangSAM · text segmentation").id == "lang_sam"
    assert reg.match("nope") is None
    assert reg.match(None) is None


def test_bad_widget_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "id: x\nname: x\nbox_key: x\n"
        "inputs:\n  - {field: images, widget: bogus_widget}\n"
        "results:\n  - {field: \"*\", visualizer: json}\n"
    )
    with pytest.raises(RegistryError, match="bogus_widget"):
        load_registry(tmp_path)


def test_flat_plus_box_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("id: x\nname: x\nbox_key: x\nflat_config: true\n"
                 "inputs: []\nresults: []\n")
    with pytest.raises(RegistryError, match="flat_config"):
        from webui.core import load_def
        load_def(p)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("id: x\nname: x\nbox_key: x\nnot_a_field: 1\n"
                 "inputs: []\nresults: []\n")
    from webui.core import load_def
    with pytest.raises(Exception, match="not_a_field"):
        load_def(p)


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "id: x\nname: x\nbox_key: x\ninputs: []\nresults: []\n")
    (tmp_path / "b.yaml").write_text(
        "id: x\nname: x\nbox_key: x\ninputs: []\nresults: []\n")
    with pytest.raises(RegistryError, match="duplicate box id"):
        load_registry(tmp_path)


def test_to_list_json_serializable(reg):
    import json
    payload = json.dumps(reg.to_list())
    assert "lang_sam" in payload
