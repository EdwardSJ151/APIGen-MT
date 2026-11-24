"""APIGen-MT phase two, native function calling and episode acceptance."""

import copy
import json
import re

from apigen_mt_blueprints import (
    DataRejected, Verdict, at_path, check_policy, difference, dump, request_data,
)
from step_by_step_models import ToolCallWithOutput, TrajectoryStep
from trajectory_replay import verify_trajectory_replay


STOP = "###STOP###"
USER = """Simulate the human described in the private task. Write one natural user message.
Stay in persona and reveal only what is needed now. Never invent facts, expose APIs or
provide facts only the assistant can discover. Paraphrase the intent. When the visible
dialogue satisfies the request, reply with exactly ###STOP###."""
SCORE = """Score the supplied candidate from 0 to 10 for fidelity to the private intent and
usefulness at this point in the dialogue. Penalize changed facts, premature disclosures,
irrelevant replies and premature stopping. Natural questions and an appropriate stop
are valid. Return only <score>INTEGER</score>."""
OUTPUT_REVIEW = """Check this completed episode against the private intent, grounded answer
obligations and policies. Require the user-facing assistant text to deliver each answer
without materially false claims. For backend facts, require supporting observations
from the live tool history before the answer. A tool result alone does not answer the user. Check all
conversation-dependent constraints, including real consent and its timing. Check that
the scenario kind was demonstrated, especially needed clarification and resumption.
For infeasible cases require a truthful explanation. Return JSON with passed (boolean)
and reason (short explanation citing turn indices)."""


def choose_user(chat, task, context, dialogue):
    view = {"q": task.q, "persona": context["persona"], "dialogue": dialogue,
            "kind": context["kind"]}
    candidates = []
    for _ in range(4):
        reply = chat([{"role": "system", "content": USER},
                      {"role": "user", "content": dump(view)}], temperature=0.8, max_tokens=180)
        candidate = (reply.get("content") or "").strip()
        if candidate and "\n" not in candidate and (STOP not in candidate or candidate == STOP):
            candidates.append(candidate)
    scores = []
    for candidate in candidates:
        reply = chat([{"role": "system", "content": SCORE},
                      {"role": "user", "content": dump({**view, "candidate": candidate})}], max_tokens=64)
        score = re.fullmatch(r"\s*<score>\s*(10|[0-9])\s*</score>\s*", reply.get("content") or "")
        if score:
            scores.append((int(score[1]), candidate))
    if not scores:
        raise DataRejected("User candidates were empty or unscorable")
    return max(scores, key=lambda pair: pair[0])[1]


class ReplayAdapter:
    """Use Minpeter's replay against a disposable clone of the same world."""

    def __init__(self, world):
        self.world = world
        self.env = world.env.clone()

    def get_api_state(self):
        return {"state": self.env.state_snapshot()}

    def restore_api_state(self, state):
        if dump(state["state"]) != dump(self.world.initial):
            raise ValueError("Replay requires the frozen original world")
        self.env = self.world.env.clone()

    def has_python_implementation(self, name):
        return name in self.world.tools

    def is_replay_safe(self, name):
        return self.has_python_implementation(name)

    def invoke_python_tool(self, name, arguments):
        return self.env.execute(name, arguments)


def check_acceptance(world, task, verified, context, transcript, trace, final, chat):
    steps = [
        TrajectoryStep(
            step_number=index,
            tool_calls=[ToolCallWithOutput(tool_name=entry["tool"], arguments=entry["arguments"],
                                           output=entry["result"])],
            pre_state={"state": entry["before"]}, post_state={"state": entry["after"]},
        )
        for index, entry in enumerate(trace, 1)
    ]
    replay = verify_trajectory_replay(ReplayAdapter(world), steps, {"state": world.initial}, exact=True)
    if not replay.is_valid:
        raise DataRejected("Tool replay differs from the recorded episode")
    actual_effects = difference(world.initial, final.state_snapshot())
    if dump(actual_effects) != dump(verified["effects"]):
        expected_groups = sorted(verified["effects"])
        actual_groups = sorted(actual_effects)
        tool_names = [entry["tool"] for entry in trace]
        raise DataRejected(f"Final state effects differ from the verified reference. "
                           f"Expected groups {expected_groups}, actual groups {actual_groups}, "
                           f"executed calls {tool_names}")
    if task.goal_spec["success"] or task.goal_spec["forbidden"]:
        if final.check_goal({"id": world.id, "goal_spec": task.goal_spec}).get("passed") is not True:
            raise DataRejected("Final state fails the environment goal")
    check_policy(world, final, trace, transcript)
    users = sum(m["role"] == "user" for m in transcript)
    batches = [m["tool_calls"] for m in transcript if m.get("tool_calls")]
    kind = context["kind"]
    if ((kind == "single_call" and (users != 1 or len(trace) != 1))
            or (kind == "parallel" and not any(len(batch) > 1 for batch in batches))
            or (kind == "multi_step" and len(batches) < 2)
            or (kind in {"multi_turn", "clarification"} and users < 2)
            or (kind == "no_tool" and trace)):
        raise DataRejected(f"Episode does not demonstrate {kind}")
    replies = [message.get("content") or "" for message in transcript
               if message["role"] == "assistant"]
    for evidence in task.evidence:
        if evidence.answer_forms and not any(
            re.search(r"(?<![\w.,])" + re.escape(form.strip()) + r"(?!\w|[.,]\d)",
                      reply, flags=re.IGNORECASE)
            for form in evidence.answer_forms if form.strip() for reply in replies
        ):
            raise DataRejected(f"Missing answer for {evidence.call.tool}{evidence.path}")
    verdict = request_data(chat, OUTPUT_REVIEW, {
        "q": task.q, "o_gt": task.o_gt, "evidence": verified["evidence"], "kind": kind,
        "transcript": transcript, "policies": world.policies,
    }, Verdict)
    if not verdict.passed:
        raise DataRejected(verdict.reason)
    return {"replay": replay.model_dump(), "output_review": verdict.model_dump()}


def execute_batch(env, world, message, messages, trace, seen):
    parsed = []
    for call in message["tool_calls"]:
        try:
            call_id = call["id"]
            name = call["function"]["name"]
            raw = call["function"]["arguments"]
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            dump(arguments)
            if (call.get("type") != "function" or not isinstance(call_id, str)
                    or not call_id or call_id in seen or name not in world.tools
                    or not isinstance(arguments, dict)):
                raise ValueError("Call ID, function, or argument object is invalid")
        except (KeyError, TypeError, ValueError) as error:
            raise DataRejected(f"Malformed native tool call: {error}") from error
        seen.add(call_id)
        parsed.append((call_id, name, arguments))

    visible_index = len(messages) - 1
    failed = []
    for call_id, name, arguments in parsed:
        before = env.state_snapshot()
        result = copy.deepcopy(env.execute(name, copy.deepcopy(arguments)))
        after = env.state_snapshot()
        entry = {"tool": name, "arguments": arguments, "result": result, "call_id": call_id,
                 "message_index": visible_index, "before": before, "after": after}
        trace.append(entry)
        if any(rule["tool"] == name for rule in world.consent):
            check_policy(world, env, trace, messages)
        messages.append({"role": "tool", "tool_call_id": call_id, "content": dump(result)})
        if result.get("ok") is not True:
            failed.append(entry)
    return failed


def expected_read_error(task, entry):
    for evidence in task.evidence:
        if (evidence.call.tool != entry["tool"]
                or dump(evidence.call.arguments) != dump(entry["arguments"])
                or not evidence.path.startswith("/error/")):
            continue
        try:
            return dump(at_path(entry["result"], evidence.path)) == dump(evidence.equals)
        except (KeyError, IndexError, TypeError, ValueError):
            return False
    return False


def rollout(world, task, verified, context, chat, student=None):
    env = world.env.clone()
    tools = world.tool_menu()
    messages = [{"role": "system", "content":
                 "Help the user using the available tools when needed. Follow these domain policies.\n"
                 + "\n".join(world.policies)}]
    trace, dialogue, supervised = [], [], []
    call_ids = set()
    awaiting_user = True
    actor = student or chat
    recovered = False
    terminated = False
    for _ in range(40):
        if awaiting_user:
            user_text = choose_user(chat, task, context, dialogue)
            if user_text == STOP:
                if not dialogue:
                    raise DataRejected("User stopped before the first message")
                terminated = True
                break
            user = {"role": "user", "content": user_text}
            messages.append(user)
            dialogue.append(user)

        response = actor(messages, tools=tools, tool_choice="auto", temperature=0.4)
        message = {"role": "assistant", "content": response.get("content")}
        if response.get("tool_calls"):
            message["tool_calls"] = response["tool_calls"]
        elif not isinstance(message["content"], str) or not message["content"].strip():
            raise DataRejected("Empty assistant response")
        if actor is chat:
            supervised.append(len(messages))
        messages.append(message)
        if message.get("tool_calls"):
            failed = execute_batch(env, world, message, messages, trace, call_ids)
            if context["kind"] == "infeasible":
                failed = [entry for entry in failed if not expected_read_error(task, entry)]
            if failed and actor is student:
                if any(
                    entry["result"].get("error", {}).get("type") not in {
                        "transient_error", "invalid_argument", "missing_argument", "not_found",
                    } or dump(entry["before"]) != dump(entry["after"])
                    for entry in failed
                ):
                    raise DataRejected("Student error is outside the recoverable set")
                check_policy(world, env, trace, messages)
                actor = chat
                recovered = True
            elif failed:
                raise DataRejected(f"Unexpected tool failures: {[(e['tool'], e['result'].get('error')) for e in failed]}")
            awaiting_user = False
        else:
            dialogue.append(message)
            awaiting_user = True

    if not terminated:
        raise DataRejected("Assistant message budget reached")
    if student is not None and not recovered:
        raise DataRejected("Student attempt did not yield a recovery example")
    checks = check_acceptance(world, task, verified, context, messages, trace, env, chat)
    return {"messages": messages, "tools": tools, "trace": trace, "supervised": supervised,
            "recovery": recovered, "verification": checks}


def sft_rows(episode):
    """Conversational prompt/completion rows for completion-only SFT."""
    messages = copy.deepcopy(episode["messages"])
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
    for target in episode["supervised"]:
        completion = messages[target]
        yield {
            "prompt": messages[:target], "completion": [completion], "tools": episode["tools"],
            "episode_id": episode["id"], "blueprint_id": episode["blueprint_id"],
            "family": episode["family"], "kind": episode["kind"],
            "target_type": "tool_call" if completion.get("tool_calls") else "text",
            "tool_call_count": len(completion.get("tool_calls", [])),
        }


def transcript_key(episode):
    messages = copy.deepcopy(episode["messages"])
    for message in messages:
        message.pop("tool_call_id", None)
        for call in message.get("tool_calls", []):
            call.pop("id", None)
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
    return dump(messages)
