"""APIGen-MT phase one, following §§4.1.1–4.1.3 and Figures 7–10."""

import copy
import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Call(Record):
    tool: str
    arguments: dict


class AnswerEvidence(Record):
    call: Call
    path: str
    equals: object
    when: str = Field(pattern="^(initial|final)$")
    answer_forms: list[str] = Field(default_factory=list)


class Blueprint(Record):
    q: str = Field(min_length=1)
    a_gt: list[Call]
    o_gt: list[str]
    evidence: list[AnswerEvidence]
    goal_spec: dict[str, list[dict]]


class Review(Record):
    reflection: str
    correctness: StrictInt = Field(ge=0, le=1)
    completeness: StrictInt = Field(ge=0, le=1)
    satisfaction: StrictInt = Field(ge=0, le=1)
    creativity: StrictInt = Field(ge=0, le=1)
    total: StrictInt = Field(ge=0, le=4)
    correction: str


class Verdict(Record):
    passed: StrictBool
    reason: str


class DataRejected(ValueError):
    """A generated candidate failed validation."""


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def request_data(chat, prompt, data, schema, tag=None):
    response = chat([{"role": "system", "content": prompt},
                     {"role": "user", "content": dump(data)}],
                    **({"max_tokens": 3072} if tag else {}))
    content = response.get("content") or ""
    if tag:
        match = re.fullmatch(r"\s*<thought>\s*(.*?)\s*</thought>\s*<" + tag
                             + r">\s*(.*?)\s*</" + tag + r">\s*", content, re.S)
        if not match or not match[1].strip():
            raise DataRejected(f"Expected a private assessment and <{tag}> payload")
        content = match[2]
    try:
        return schema.model_validate_json(content)
    except ValidationError as error:
        raise DataRejected(str(error)) from error


@dataclass
class World:
    id: str
    family: str
    seed: int
    env: object
    policies: list[str]
    task_rules: list[str]
    records: list[dict]
    personas: list[str]
    edges: list[list[str]]
    examples: list[dict]
    kinds: list[str]
    consent: list[dict] = field(default_factory=list)
    output_examples: dict = field(default_factory=dict)

    def __post_init__(self):
        self.env.reset(self.seed, self.id)
        self.tools = {tool["name"]: tool for tool in self.env.tool_specs()}
        self.initial = self.env.state_snapshot()
        for source, target in self.edges:
            if source not in self.tools or target not in self.tools:
                raise ValueError(f"Unknown dependency in {self.id}: {source} -> {target}")
        if not self.records or not self.personas or not self.kinds or not self.tools:
            raise ValueError(f"{self.id} needs records, personas, scenario kinds, and tools")

    def tool_menu(self):
        return [
            {"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
            for t in self.tools.values()
        ]


def sample_context(world, rng, kind):
    names = list(world.tools)
    starts = names
    if kind == "policy":
        guarded = {rule["tool"] for rule in world.consent}
        starts = [name for name in names if name in guarded] or names
    elif kind == "single_call":
        reads = [name for name in names if world.tools[name]["class"] == "read"]
        starts = ([name for name in reads
                   if world.output_examples.get(name, {}).get("ok") is True] or reads or names)
    elif kind == "parallel":
        names = [name for name in names if world.tools[name]["class"] == "read"]
        if len(names) < 2:
            raise DataRejected("Parallel sampling requires at least two read operations")
        starts = names
    selected = [rng.choice(starts)]
    if kind == "parallel":
        neighbours = {b for a, b in world.edges if a == selected[0]}
        neighbours.update(a for a, b in world.edges if b == selected[0])
        choices = [n for n in names if n != selected[0] and n not in neighbours]
        if not choices:
            raise DataRejected("No independent tool pair")
        selected.append(rng.choice(choices))
    elif kind != "single_call":
        for _ in range(rng.randint(0, 4)):
            successors = sorted({b for a, b in world.edges if a == selected[-1]} - set(selected))
            if not successors:
                break
            selected.append(rng.choice(successors))
    examples = list(world.examples)
    rng.shuffle(examples)
    examples.sort(key=lambda example: -len(set(selected) & set(example.get("tools", []))))
    record = rng.choice(world.records)
    return {
        "tools": [world.tools[name] for name in selected],
        "policies": world.policies,
        "sampled_policy_focus": rng.sample(world.policies, rng.randint(0, min(3, len(world.policies)))),
        "task_rules": world.task_rules,
        "records": record,
        "tool_output_examples": world.output_examples,
        "known_goal_facts": world.env.known_facts() if hasattr(world.env, "known_facts") else [],
        "persona": rng.choice(world.personas),
        "examples": examples[:rng.randint(1, 3)],
        "kind": kind,
    }


def at_path(value, path):
    """Resolve a JSON Pointer against an executed reference observation."""
    if not path:
        return value
    if not path.startswith("/"):
        raise KeyError(path)
    for component in path[1:].split("/"):
        key = component.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def resolve(value, observations):
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            index, _, path = value["$ref"].partition("#")
            if not index.isdecimal() or not 1 <= int(index) <= len(observations):
                raise KeyError(value["$ref"])
            return at_path(observations[int(index) - 1], path)
        return {k: resolve(v, observations) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(item, observations) for item in value]
    return value


def difference(initial, final):
    return {key: {"before": initial.get(key), "after": final.get(key)}
            for key in sorted(initial.keys() | final.keys())
            if dump(initial.get(key)) != dump(final.get(key)) or
            (key in initial) != (key in final)}


def check_policy(world, env, trace, messages=None):
    policy = env.check_policy([{k: call[k] for k in ("tool", "arguments", "result")} for call in trace])
    if policy.get("passed") is not True:
        raise DataRejected(f"Policy failure: {dump(policy)}")
    if messages is None:
        return
    for call in trace:
        for rule in world.consent:
            if call["tool"] != rule["tool"]:
                continue
            statement = call["arguments"].get(rule["argument"], "")
            prior = messages[:call["message_index"]]
            if not isinstance(statement, str) or not statement.strip() or not any(
                m["role"] == "user" and statement.strip() in m["content"]
                for m in prior
            ):
                raise DataRejected("Authorization statement lacks an earlier user utterance")


def execute_reference(world, task):
    env = world.env.clone()
    trace = []
    for index, call in enumerate(task.a_gt, 1):
        if world.tools.get(call.tool, {}).get("class") != "write":
            raise DataRejected(f"Reference action {index} must name a write operation")
        try:
            arguments = resolve(call.arguments, [entry["result"] for entry in trace])
            result = copy.deepcopy(env.execute(call.tool, copy.deepcopy(arguments)))
        except (KeyError, TypeError, IndexError, ValueError, AttributeError) as error:
            raise DataRejected(f"Reference action {index}: {error}") from error
        if result.get("ok") is not True:
            raise DataRejected(f"Reference action {index}: {dump(result)}")
        trace.append({"tool": call.tool, "arguments": arguments, "result": result})

    check_policy(world, env, trace)
    if set(task.goal_spec) != {"success", "forbidden"}:
        raise DataRejected("goal_spec requires success and forbidden lists")
    if task.goal_spec["success"] or task.goal_spec["forbidden"]:
        goal = env.check_goal({"id": world.id, "goal_spec": task.goal_spec})
        if goal.get("passed") is not True or any(
            isinstance(item.get("actual"), str) and item["actual"].startswith(
                ("unknown_fact:", "unknown_op:", "bad_args:"))
            for item in goal.get("success_results", []) + goal.get("forbidden_results", [])
        ):
            raise DataRejected(f"Reference fails declared goal: {dump(goal)}")
    effects = difference(world.initial, env.state_snapshot())
    if not effects and not task.o_gt:
        raise DataRejected("Task has neither an observable effect nor an answer obligation")
    facts = []
    for evidence in task.evidence:
        if world.tools.get(evidence.call.tool, {}).get("class") != "read":
            raise DataRejected("Answer evidence requires an executable read tool")
        source = world.env if evidence.when == "initial" else env
        result = source.clone().execute(evidence.call.tool, evidence.call.arguments)
        try:
            if dump(at_path(result, evidence.path)) != dump(evidence.equals):
                raise DataRejected(f"Evidence mismatch for {evidence.call.tool}{evidence.path}")
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise DataRejected(f"Invalid evidence for {evidence.call.tool}: {error}") from error
        facts.append(evidence.model_dump())
    return {"effects": effects, "reference_trace": trace, "evidence": facts}


WRITER = """Write one realistic task in the supplied persona and world. Use examples for format
and variety; invent no fixture facts and avoid copying example intents or call patterns.
q is the complete private user intent and only facts the user knows, including genuine
information gaps or consent conditions. a_gt lists only state-changing reference calls.
Use {"$ref":"1#/result/field"} for the actual result of an earlier 1-indexed call.
o_gt lists specific information owed to the user; omit read-only discovery calls from a_gt.
Ground API-derived answers using evidence entries with call, JSON Pointer path, equals and
when (initial or final). Keep the expected answer out of q if the user does not know it.
For no-tool tasks give a directly checkable response obligation. For infeasible tasks give
the specific limitation and supporting evidence when it depends on state.
The default goal_spec is {"success":[],"forbidden":[]}; the executor always compares the
reference effects. Only use a named goal predicate when its name and arguments appear
in a provided example. Never invent goal predicates.
Write one short sentence inside <thought>, followed immediately by one strict JSON
object inside <answer> containing q, a_gt, o_gt, evidence, and goal_spec. Use the actual
response paths shown in tool_output_examples. Put no Markdown fences around the JSON.
For a policy scenario prefer an executable task that requires following a rule.
A single_call task is fully specified in the opening request. A parallel task needs two
independent operations. A multi_step task has a dependency between calls. A multi_turn
task reveals distinct subgoals over multiple user turns. A clarification task withholds
a fact the user knows and the assistant cannot safely infer. An infeasible task needs a
grounded explanation. A no_tool task should be answerable without any execution.
For a specific quantity owed to the user, include equivalent answer_forms as literal
strings, e.g. ["$149.00", "14,900 cents", "14900 cents"]; omit for open-ended evidence.
On revision fix the supplied feedback directly and keep the private assessment to one sentence."""

REVIEWER = """Review this task and its actual effects independently. Return JSON with reflection,
correctness, completeness, satisfaction, creativity, total, correction. Each criterion is
0 or 1 and total is their sum. Correctness requires lawful intended effects with no extras.
Completeness requires enough private intent and an achievable solution using visible tools.
Satisfaction means the expected answers cover the request and are supported by the records
or observations. Creativity means a plausible, varied scenario appropriate to the requested
kind, including simple tasks. Flag leaked backend facts, copied examples, and guessable
clarification. For parallel tasks require independently executable calls."""

SUMMARIZER = """Synthesize the existing reviewer assessments. Identify agreement and
disagreement and give the writer a short actionable revision. Use the task and actual
effects to ground the feedback. Return only the summary."""


def review_task(world, task, verified, context, chat):
    review_input = {
        "task": task.model_dump(), "tools": list(world.tools.values()),
        "policies": world.policies, "kind": context["kind"],
        "diff_patch": verified["effects"], "evidence": verified["evidence"],
        "examples": context["examples"],
        "records": context.get("records", {}),
    }
    criteria = ("correctness", "completeness", "satisfaction", "creativity")
    votes = []
    for _ in range(3):
        try:
            vote = request_data(chat, REVIEWER, review_input, Review).model_dump()
            if vote["total"] != sum(vote[key] for key in criteria):
                raise DataRejected("Inconsistent review total")
        except DataRejected as error:
            vote = {"reflection": str(error), **dict.fromkeys(criteria, 0)}
        votes.append(vote)
    if all(sum(vote[key] for vote in votes) >= 2 for key in criteria):
        return votes
    summary = chat([{"role": "system", "content": SUMMARIZER},
                    {"role": "user", "content": dump({**review_input, "reviews": votes})}])
    raise DataRejected(summary.get("content") or dump(votes))


def generate_blueprint(world, context, chat):
    feedback = ""
    previous = None
    for _ in range(4):
        try:
            task = request_data(chat, WRITER, {**context, "feedback": feedback,
                                "previous_draft": previous}, Blueprint, "answer")
            previous = task.model_dump()
            if context["kind"] == "single_call":
                sampled_tool = context["tools"][0]
                if sampled_tool["class"] == "read":
                    if task.a_gt or not task.evidence or any(
                        item.call.tool != sampled_tool["name"] for item in task.evidence
                    ):
                        raise DataRejected("Use only the sampled read tool for this one-call task")
                elif len(task.a_gt) != 1 or task.a_gt[0].tool != sampled_tool["name"]:
                    raise DataRejected("Use only the sampled write tool for this one-call task")
            if context["kind"] in {"no_tool", "infeasible"} and task.a_gt:
                raise DataRejected("No-tool and infeasible cases require no reference writes")
            if context["kind"] in {"single_call", "parallel", "multi_step", "multi_turn",
                                   "clarification", "policy", "recovery"} and not (
                                       task.a_gt or task.evidence):
                raise DataRejected("This scenario needs executable evidence")
            verified = execute_reference(world, task)
            return task, verified, review_task(world, task, verified, context, chat)
        except DataRejected as error:
            feedback = str(error)
    raise DataRejected(feedback)


class CombinedIntent(Record):
    q: str = Field(min_length=1)


def recombine(world, parents, context, chat):
    actions, answers, evidence = [], [], []
    goal = {"success": [], "forbidden": []}

    def shift_refs(value, offset):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                index, mark, path = value["$ref"].partition("#")
                return {"$ref": f"{int(index) + offset}{mark}{path}"}
            return {k: shift_refs(v, offset) for k, v in value.items()}
        if isinstance(value, list):
            return [shift_refs(v, offset) for v in value]
        return value

    for parent in parents:
        offset = len(actions)
        actions.extend(Call(tool=call.tool, arguments=shift_refs(call.arguments, offset))
                       for call in parent.a_gt)
        answers.extend(parent.o_gt)
        evidence.extend(parent.evidence)
        for kind in goal:
            goal[kind].extend(parent.goal_spec[kind])
    merged = Blueprint(q=" / ".join(p.q for p in parents), a_gt=actions, o_gt=answers,
                       evidence=evidence, goal_spec=goal)
    verified = execute_reference(world, merged)
    feedback = ""
    for _ in range(4):
        try:
            intent = request_data(chat, "Combine the two user intents into one coherent private "
                                  "intent with the same goals and disclosure constraints. "
                                  "Return JSON with q.", {
                                      "persona": context["persona"],
                                      "intents": [parent.q for parent in parents],
                                      "feedback": feedback,
                                  }, CombinedIntent)
            task = merged.model_copy(update={"q": intent.q})
            return task, verified, review_task(world, task, verified, context, chat)
        except DataRejected as error:
            feedback = str(error)
    raise DataRejected(feedback)
