#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import entrypoint


WORKFLOW_NAME = "infinitetalk-v2v-latentsync16-stable-docker.json"
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "defaults/workflows"
    / WORKFLOW_NAME
)


def workflow_without_flashvsr(source: dict) -> dict:
    workflow = copy.deepcopy(source)
    workflow["nodes"] = [
        node
        for node in workflow["nodes"]
        if node["id"] not in {308, 309, 310, 311, 312}
    ]
    workflow["links"] = [link for link in workflow["links"] if link[0] <= 560]
    valid_links = {link[0] for link in workflow["links"]}
    for node in workflow["nodes"]:
        for output in node.get("outputs", []):
            links = output.get("links")
            if isinstance(links, list):
                output["links"] = [link for link in links if link in valid_links]
    workflow["last_node_id"] = 307
    workflow["last_link_id"] = 560
    workflow.setdefault("extra", {}).pop("infinitetalk_flashvsr_schema", None)
    return workflow


def remap_ids(workflow: dict, node_offset: int, link_offset: int) -> dict:
    node_ids = {node["id"]: node["id"] + node_offset for node in workflow["nodes"]}
    link_ids = {link[0]: link[0] + link_offset for link in workflow["links"]}
    for node in workflow["nodes"]:
        node["id"] = node_ids[node["id"]]
        for input_value in node.get("inputs", []):
            if input_value.get("link") in link_ids:
                input_value["link"] = link_ids[input_value["link"]]
        for output in node.get("outputs", []):
            if isinstance(output.get("links"), list):
                output["links"] = [link_ids[link] for link in output["links"]]
    for link in workflow["links"]:
        link[0] = link_ids[link[0]]
        link[1] = node_ids[link[1]]
        link[3] = node_ids[link[3]]
    workflow["last_node_id"] += node_offset
    workflow["last_link_id"] += link_offset
    return workflow


def workflow_schema1(source: dict) -> dict:
    workflow = workflow_without_flashvsr(source)
    source_nodes = {node["id"]: node for node in source["nodes"]}
    for node_id in (308, 309, 310):
        workflow["nodes"].append(copy.deepcopy(source_nodes[node_id]))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    nodes[301]["outputs"][0]["links"].append(561)
    nodes[254]["outputs"][0]["links"].append(564)
    nodes[308]["inputs"][0]["link"] = 561
    nodes[308]["outputs"][0]["links"] = [562]
    nodes[309]["inputs"][0]["link"] = 562
    nodes[309]["outputs"][0]["links"] = [563]
    nodes[310]["inputs"][0]["link"] = 563
    nodes[310]["inputs"][1]["link"] = 564
    workflow["links"].extend(
        (
            [561, 301, 0, 308, 0, "IMAGE"],
            [562, 308, 0, 309, 0, "IMAGE"],
            [563, 309, 0, 310, 0, "IMAGE"],
            [564, 254, 0, 310, 1, "AUDIO"],
        )
    )
    workflow["last_node_id"] = 310
    workflow["last_link_id"] = 564
    workflow["extra"]["infinitetalk_flashvsr_schema"] = 1
    return workflow


def combine_by_prefix(workflow: dict, prefix: str) -> dict:
    return next(
        node
        for node in workflow["nodes"]
        if node.get("type") == "VHS_VideoCombine"
        and node.get("widgets_values", {}).get("filename_prefix") == prefix
    )


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    target = remap_ids(workflow_without_flashvsr(source), 1000, 2000)
    original_max_node = max(node["id"] for node in target["nodes"])
    original_max_link = max(link[0] for link in target["links"])

    with tempfile.TemporaryDirectory() as temporary:
        comfyui_home = Path(temporary)
        workflow_dir = comfyui_home / "user/default/workflows"
        workflow_dir.mkdir(parents=True)
        target_path = workflow_dir / WORKFLOW_NAME
        target_path.write_text(json.dumps(target), encoding="utf-8")
        entrypoint.COMFYUI_HOME = comfyui_home

        entrypoint.upgrade_stable_flashvsr(SOURCE, WORKFLOW_NAME)
        migrated = json.loads(target_path.read_text(encoding="utf-8"))
        if migrated.get("extra", {}).get("infinitetalk_flashvsr_schema") != 2:
            raise SystemExit("Marcador da migracao FlashVSR nao foi aplicado")
        if sum(
            node.get("type") == "AILab_FlashVSR_Advanced"
            for node in migrated["nodes"]
        ) != 1:
            raise SystemExit("A migracao deve adicionar exatamente um FlashVSR")
        if sum(
            node.get("type") == "VHS_SelectFilename" for node in migrated["nodes"]
        ) != 1 or sum(
            node.get("type") == "VHS_LoadVideoPath" for node in migrated["nodes"]
        ) != 1:
            raise SystemExit("A migracao deve recarregar o MP4 Stable salvo")
        if migrated["last_node_id"] <= original_max_node:
            raise SystemExit("A migracao FlashVSR colidiu com IDs de nodes")
        if migrated["last_link_id"] <= original_max_link:
            raise SystemExit("A migracao FlashVSR colidiu com IDs de links")

        links = {link[0]: link for link in migrated["links"]}
        base = combine_by_prefix(
            migrated, "InfiniteTalk_V2V_LatentSync16_Stable"
        )
        fullhd = combine_by_prefix(
            migrated, "InfiniteTalk_V2V_LatentSync16_Stable_FullHD"
        )
        base_audio = links[base["inputs"][1]["link"]]
        fullhd_audio = links[fullhd["inputs"][1]["link"]]
        if base_audio[1:3] != fullhd_audio[1:3]:
            raise SystemExit("A migracao nao preservou o audio original")
        flash = next(
            node
            for node in migrated["nodes"]
            if node.get("type") == "AILab_FlashVSR_Advanced"
        )
        flash_input = links[flash["inputs"][0]["link"]]
        flash_source = next(
            node for node in migrated["nodes"] if node["id"] == flash_input[1]
        )
        if flash_source.get("type") != "VHS_LoadVideoPath":
            raise SystemExit("FlashVSR nao recebe o MP4 Stable recarregado")

        before_second_run = copy.deepcopy(migrated)
        entrypoint.upgrade_stable_flashvsr(SOURCE, WORKFLOW_NAME)
        after_second_run = json.loads(target_path.read_text(encoding="utf-8"))
        if after_second_run != before_second_run:
            raise SystemExit("A migracao FlashVSR nao e idempotente")

    persisted_schema1 = remap_ids(workflow_schema1(source), 3000, 4000)
    with tempfile.TemporaryDirectory() as temporary:
        comfyui_home = Path(temporary)
        workflow_dir = comfyui_home / "user/default/workflows"
        workflow_dir.mkdir(parents=True)
        target_path = workflow_dir / WORKFLOW_NAME
        target_path.write_text(json.dumps(persisted_schema1), encoding="utf-8")
        entrypoint.COMFYUI_HOME = comfyui_home
        entrypoint.upgrade_stable_flashvsr(SOURCE, WORKFLOW_NAME)
        migrated = json.loads(target_path.read_text(encoding="utf-8"))
        links = {link[0]: link for link in migrated["links"]}
        flash = next(
            node
            for node in migrated["nodes"]
            if node.get("type") == "AILab_FlashVSR_Advanced"
        )
        flash_input = links[flash["inputs"][0]["link"]]
        source_node = next(
            node for node in migrated["nodes"] if node["id"] == flash_input[1]
        )
        if source_node.get("type") != "VHS_LoadVideoPath":
            raise SystemExit("Workflow schema 1 nao foi migrado para MP4 salvo")
        if migrated["extra"].get("infinitetalk_flashvsr_schema") != 2:
            raise SystemExit("Workflow schema 1 nao recebeu o schema 2")

    print("OK: migracao FlashVSR por arquivo dinamica e idempotente")


if __name__ == "__main__":
    main()
