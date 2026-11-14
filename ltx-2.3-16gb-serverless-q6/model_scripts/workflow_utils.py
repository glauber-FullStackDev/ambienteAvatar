#!/usr/bin/env python3
"""Utilitarios compartilhados de manipulacao de workflows ComfyUI 0.4."""
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def one(
    items: list[dict],
    *,
    node_id: int | None = None,
    node_type: str | None = None,
) -> dict:
    matches = [
        item
        for item in items
        if (node_id is None or item.get("id") == node_id)
        and (node_type is None or item.get("type") == node_type)
    ]
    if len(matches) != 1:
        wanted = f"id={node_id}, type={node_type}"
        raise ValueError(
            f"esperado exatamente um node ({wanted}); encontrados {len(matches)}"
        )
    return matches[0]


def link_map(subgraph: dict) -> dict[int, dict]:
    return {link["id"]: link for link in subgraph["links"]}


def add_link(
    subgraph: dict,
    link_id: int,
    origin_id: int | str,
    origin_slot: int,
    target_id: int | str,
    target_slot: int,
    link_type: str,
) -> None:
    subgraph["links"].append(
        {
            "id": link_id,
            "origin_id": origin_id,
            "origin_slot": origin_slot,
            "target_id": target_id,
            "target_slot": target_slot,
            "type": link_type,
        }
    )


def replace_input_link(node: dict, input_name: str, link_id: int | None) -> None:
    target = one([item for item in node["inputs"] if item.get("name") == input_name])
    target["link"] = link_id


def replace_output_links(node: dict, output_name: str, links: list[int]) -> None:
    target = one([item for item in node["outputs"] if item.get("name") == output_name])
    target["links"] = links


def next_node_id(items: list[dict]) -> int:
    return max(item["id"] for item in items if isinstance(item.get("id"), int)) + 1
