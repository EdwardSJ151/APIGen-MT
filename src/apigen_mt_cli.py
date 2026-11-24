"""Command-line entry point for source-grounded APIGen-MT synthesis."""

import argparse
import importlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import requests

from apigen_mt_blueprints import (
    DataRejected, World, dump, generate_blueprint,
    recombine, sample_context,
)
from apigen_mt_rollouts import rollout, sft_rows, transcript_key
from generate_step_by_step import _append_datapoint
from llm_request_helpers import retry_delay


STRATA_URL = os.environ.get("STRATA_CHAT_URL")
KINDS = {"single_call", "parallel", "multi_step", "multi_turn", "clarification",
         "no_tool", "infeasible", "policy", "recovery"}


class Chat:
    def __init__(self, url, model, api_key, strata=False):
        self.url = url
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        if strata:
            self.headers["ngrok-skip-browser-warning"] = "true"
        self.strata = strata
        self.models = set()
        self.tokens = Counter()
        self.calls = 0

    def __call__(self, messages, **options):
        payload = {"model": self.model, "messages": messages, "max_tokens": 1536,
                   "temperature": 0.3, **options}
        if self.strata:
            payload["reasoning_effort"] = "none"
        for attempt in range(3):
            try:
                response = requests.post(self.url, headers=self.headers, json=payload,
                                         timeout=(10, 90), allow_redirects=False)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == 2:
                    raise RuntimeError("Model endpoint connection failed") from None
            else:
                if response.status_code == 200:
                    try:
                        result = response.json()
                    except ValueError as error:
                        raise RuntimeError("Invalid JSON from chat completion endpoint") from error
                    if not isinstance(result, dict):
                        raise RuntimeError("Invalid chat completion payload")
                    if result.get("model"):
                        self.models.add(result["model"])
                    self.calls += 1
                    self.tokens.update({name: value for name, value in result.get("usage", {}).items()
                                        if name.endswith("_tokens") and isinstance(value, int)})
                    try:
                        choice = result["choices"][0]
                        if choice.get("finish_reason") in {"length", "content_filter"}:
                            return {}
                        return choice["message"]
                    except (KeyError, IndexError, TypeError) as error:
                        raise RuntimeError("Malformed chat completion response") from error
                if response.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise RuntimeError(f"Chat completion returned HTTP {response.status_code}")
            time.sleep(retry_delay(2, attempt, 8, 0.25, random.uniform))
        raise RuntimeError("Chat completion retry budget exhausted")


def load_worlds(path):
    manifest = json.loads(path.read_text())
    if package_root := manifest.get("python_path"):
        sys.path.append(str((path.parent / package_root).resolve()))
    held_out = set(manifest.get("holdout_families", []))
    ids = set()
    worlds = []
    for spec in manifest["worlds"]:
        spec = dict(spec)
        if spec["id"] in ids:
            raise ValueError(f"Duplicate world ID {spec['id']}")
        ids.add(spec["id"])
        if spec["family"] in held_out:
            continue
        module, _, name = spec.pop("factory").partition(":")
        kwargs = spec.pop("factory_args", {})
        if fixture := spec.pop("fixture", None):
            kwargs["fixture"] = json.loads((path.parent / fixture).read_text())
        factory = getattr(importlib.import_module(module), name)
        if set(spec["kinds"]) - KINDS:
            raise ValueError(f"Unrecognized scenario kind for {spec['id']}")
        worlds.append(World(env=factory(**kwargs), **spec))
    if not worlds:
        raise ValueError("Manifest has no training worlds after family holdout")
    return worlds


def generate(worlds, chat, student, output, rounds, seed, report=None):
    rng = random.Random(seed)
    summary = Counter()
    by_world = {world.id: Counter() for world in worlds}
    parents = {}
    seen_blueprints = set()

    def append(file, values):
        _append_datapoint(str(output / (file + ".jsonl")), values)

    def reject(world, stage, error, **identity):
        summary[f"{stage}_rejected"] += 1
        by_world[world.id][f"{stage}_rejected"] += 1
        append("rejections", {"world_id": world.id, "stage": stage,
                              "reason": str(error), **identity})

    def collect(world, blueprint, evidence, reviews, context, parent_ids):
        identity = dump([world.id, context["persona"], blueprint.model_dump()])
        if identity in seen_blueprints:
            reject(world, "blueprint", "Duplicate validated blueprint")
            return None
        seen_blueprints.add(identity)
        summary["blueprints"] += 1
        by_world[world.id]["blueprints"] += 1
        blueprint_id = f"{world.id}-{summary['blueprints']:06d}"
        task = blueprint.model_dump()
        append("blueprints", {"id": blueprint_id, "world_id": world.id, "family": world.family,
                              "seed": world.seed, "kind": context["kind"],
                              "persona": context["persona"], "parents": parent_ids,
                              "task": task, "checks": evidence, "votes": reviews})
        world.examples.append({"task": task, "tools": list(dict.fromkeys(
            [step.tool for step in blueprint.a_gt] + [item.call.tool for item in blueprint.evidence]))})
        seen = set()
        for attempt in range(1, 4):
            recovery_actor = student if context["kind"] == "recovery" and attempt == 3 else None
            if context["kind"] == "recovery" and attempt == 3 and not student:
                summary["recovery_attempts_skipped"] += 1
                continue
            rollout_context = context if recovery_actor or context["kind"] != "recovery" else {
                **context, "kind": "standard",
            }
            summary["rollout_attempts"] += 1
            by_world[world.id]["rollout_attempts"] += 1
            try:
                episode = rollout(world, blueprint, evidence, rollout_context, chat, recovery_actor)
                key = transcript_key(episode)
                if key in seen:
                    raise DataRejected("Duplicate transcript")
            except DataRejected as error:
                reject(world, "rollout", error, blueprint_id=blueprint_id, attempt=attempt)
                continue
            seen.add(key)
            episode.update(id=f"{blueprint_id}-{attempt}", blueprint_id=blueprint_id,
                           family=world.family, world_id=world.id,
                           kind=rollout_context["kind"])
            append("episodes", episode)
            for row in sft_rows(episode):
                append("sft", row)
                summary["sft_rows"] += 1
            counts = {"episodes": 1,
                      "user_turns": sum(message["role"] == "user" for message in episode["messages"]),
                      "assistant_turns": sum(message["role"] == "assistant" for message in episode["messages"]),
                      "tool_calls": len(episode["trace"]), "recoveries": int(episode["recovery"])}
            summary.update(counts)
            by_world[world.id].update(counts)
        return blueprint_id

    # Each round visits each world, with kinds cycled independently for each world.
    for round_index in range(rounds):
        for world in worlds:
            summary["blueprint_attempts"] += 1
            by_world[world.id]["blueprint_attempts"] += 1
            kind = world.kinds[round_index % len(world.kinds)]
            try:
                context = sample_context(world, rng, kind)
                blueprint, evidence, reviews = generate_blueprint(world, context, chat)
            except DataRejected as error:
                reject(world, "blueprint", error)
                continue
            identifier = collect(world, blueprint, evidence, reviews, context, [])
            if identifier is None or kind in {"no_tool", "infeasible", "recovery"}:
                continue
            group = (world.id, context["persona"], dump(context["records"]))
            pool = parents.setdefault(group, [])
            if pool:
                first_id, first = pool[-1]
                pair = (first_id, identifier)
                combined_context = {**context, "kind": "recombined"}
                summary["recombination_attempts"] += 1
                try:
                    joined, effects, votes = recombine(
                        world, [first, blueprint], combined_context, chat,
                    )
                except DataRejected as error:
                    reject(world, "recombination", error, parents=list(pair))
                else:
                    if collect(world, joined, effects, votes, combined_context, list(pair)):
                        summary["recombined_blueprints"] += 1
            pool.append((identifier, blueprint))
        if report:
            report(summary, by_world)
    return summary, by_world


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--rounds", type=int, default=8, help="Blueprint proposals per world")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=sorted(KINDS), help="Pilot one scenario kind")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if not (key := os.environ.get("STRATA_API_KEY")):
        parser.error("STRATA_API_KEY is required")
    if not (teacher_url := os.environ.get("STRATA_CHAT_URL")):
        parser.error("STRATA_CHAT_URL is required")
    worlds = load_worlds(args.manifest.resolve())
    if args.kind:
        worlds = [world for world in worlds if args.kind in world.kinds]
        if not worlds:
            parser.error("No training world supports the requested scenario kind")
        for world in worlds:
            world.kinds = [args.kind]
    teacher = Chat(teacher_url, "strata", key, strata=True)
    student_url = os.environ.get("STUDENT_CHAT_URL")
    student = (Chat(student_url, os.environ.get("STUDENT_MODEL", "Qwen/Qwen3.5-4B"),
                    os.environ.get("STUDENT_API_KEY")) if student_url else None)
    args.output.mkdir(parents=True, exist_ok=False)

    def report(summary, by_world):
        record = {"counts": summary, "by_world": by_world, "seed": args.seed,
                  "manifest": str(args.manifest.resolve()), "kind": args.kind,
                  "teacher_models": sorted(teacher.models),
                  "teacher_calls": teacher.calls, "teacher_tokens": teacher.tokens,
                  "student_models": sorted(student.models) if student else [],
                  "review": "three same-model samples, per-criterion majority"}
        (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(dump(summary), flush=True)

    summary, _ = generate(worlds, teacher, student, args.output, args.rounds, args.seed, report)
    return 0 if summary["episodes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
