"""Regression tests using the project's executable commerce environment."""

import copy
import json
from pathlib import Path

import pytest

from apigen_mt_blueprints import Blueprint, DataRejected, execute_reference, recombine
from apigen_mt_cli import generate, load_worlds
from apigen_mt_rollouts import (
    check_acceptance, rollout, sft_rows, transcript_key,
)


ROOT = Path(__file__).resolve().parents[2]
ORDER = {"order_id": "O-1001"}
APPROVAL = "I authorize cancellation of this order."
TASK = {
    "q": "Ask for the charge on order O-1001, then approve cancellation with no refund.",
    "a_gt": [
        {"tool": "request_customer_authorization",
         "arguments": {**ORDER, "action": "cancel", "user_statement": APPROVAL}},
        {"tool": "cancel_order",
         "arguments": {**ORDER, "authorization_token": {"$ref": "1#/result/authorization_token"}}},
    ],
    "o_gt": ["The charge was 14900 cents."],
    "evidence": [{"call": {"tool": "get_payment", "arguments": ORDER},
                  "path": "/result/payment/amount_cents", "equals": 14900, "when": "initial"}],
    "goal_spec": {"success": [{"fact": "order_status", "args": ORDER, "value": "cancelled"}],
                  "forbidden": []},
}


def calls(name, arguments, *additional):
    return {"content": None, "tool_calls": [
        {"id": "call-" + tool, "type": "function",
         "function": {"name": tool, "arguments": json.dumps(args)}}
        for tool, args in ((name, arguments), *additional)
    ]}


class Teacher:
    def __init__(self, answer="You paid $149.00. Do you approve cancellation?", forged=False):
        self.answer = answer
        self.forged = forged

    def __call__(self, messages, **options):
        directive = messages[0]["content"]
        if "tools" in options:
            assert "a_gt" not in json.dumps(messages)
            last = messages[-1]
            if last["role"] == "user":
                if last["content"] == APPROVAL or self.forged:
                    return calls("request_customer_authorization",
                                 {**ORDER, "action": "cancel", "user_statement": APPROVAL})
                return calls("get_order", ORDER, ("get_payment", ORDER))
            tool = last["tool_call_id"].removeprefix("call-")
            if tool == "get_payment":
                return {"content": self.answer}
            if tool == "request_customer_authorization":
                token = json.loads(last["content"])["result"]["authorization_token"]
                return calls("cancel_order", {**ORDER, "authorization_token": token})
            assert tool == "cancel_order"
            return {"content": "Your order is cancelled."}
        payload = json.loads(messages[1]["content"])
        if directive.startswith("Write one realistic task"):
            return {"content": "<thought>Check the fixture.</thought><answer>"
                               + json.dumps(TASK) + "</answer>"}
        if directive.startswith("Review this task"):
            return {"content": json.dumps({"reflection": "Valid", "correctness": 1,
                    "completeness": 1, "satisfaction": 1, "creativity": 1,
                    "total": 4, "correction": ""})}
        if directive.startswith("Simulate"):
            assert set(payload) == {"q", "persona", "dialogue", "kind"}
            return {"content": {0: "What did I pay for order O-1001?", 2: APPROVAL,
                                4: "###STOP###"}.get(len(payload["dialogue"]), "###STOP###")}
        if directive.startswith("Score"):
            return {"content": "<score>10</score>"}
        assert directive.startswith("Check this completed episode")
        replies = [m.get("content") or "" for m in payload["transcript"] if m["role"] == "assistant"]
        passed = any("$149.00" in reply for reply in replies)
        return {"content": json.dumps({"passed": passed, "reason": "Incorrect charge" if not passed else "Verified"})}


@pytest.fixture
def world():
    return load_worlds(ROOT / "examples" / "commerce-pilot.json")[0]


def context(world, kind="policy"):
    return {"persona": world.personas[0], "kind": kind, "examples": []}


def test_phase_one_and_two_generate_verified_sft(world, tmp_path):
    counts, _ = generate([world], Teacher(), None, tmp_path, 1, 0)
    assert counts["blueprints"] == 1
    assert counts["episodes"] == 1
    assert counts["rollout_attempts"] == 3
    assert len(world.examples) == 3
    episode = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert [step["tool"] for step in episode["trace"]] == [
        "get_order", "get_payment", "request_customer_authorization", "cancel_order",
    ]
    assert episode["verification"]["replay"]["status"] == "verified"
    rows = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert len(rows[0]["completion"][0]["tool_calls"]) == 2
    assert rows[0]["completion"][0]["tool_calls"][1]["function"]["arguments"] == ORDER
    assert all(len(row["completion"]) == 1 and row["completion"][0]["role"] == "assistant"
               for row in rows)
    assert '"a_gt"' not in json.dumps(rows)
    assert "###STOP###" not in json.dumps(rows)
    assert world.env.state_snapshot() == world.initial


def test_effects_consent_and_tampering(world):
    task = Blueprint.model_validate(TASK)
    verified = execute_reference(world, task)
    with pytest.raises(DataRejected, match="Incorrect charge"):
        rollout(world, task, verified, context(world), Teacher(answer="You paid $19.00."))
    with pytest.raises(DataRejected, match="earlier user utterance"):
        rollout(world, task, verified, context(world), Teacher(forged=True))
    valid = rollout(world, task, verified, context(world), Teacher())
    trace = copy.deepcopy(valid["trace"])
    final = world.env.clone()
    for entry in trace:
        final.execute(entry["tool"], entry["arguments"])
    trace[0]["result"]["result"]["order"]["status"] = "shipped"
    with pytest.raises(DataRejected, match="replay"):
        check_acceptance(world, task, verified, context(world), valid["messages"], trace, final, Teacher())
    with pytest.raises(DataRejected, match="Final state"):
        check_acceptance(world, task, verified, context(world), valid["messages"], valid["trace"],
                         world.env, Teacher())


def test_recombination_checks_conflicts_before_calling_writer(world):
    task = Blueprint.model_validate(TASK)

    def unexpected_writer(*args, **kwargs):
        pytest.fail("Incompatible tasks reached the combined intent writer")

    with pytest.raises(DataRejected, match="Reference action 4"):
        recombine(world, [task, task], context(world, "recombined"), unexpected_writer)


def test_recovery_context_has_no_student_loss(world):
    task = Blueprint.model_validate(TASK)

    def student(*args, **kwargs):
        return calls("get_payment", {})

    class RepairingTeacher(Teacher):
        def __call__(self, messages, **options):
            if "tools" in options and messages[-1].get("tool_call_id") == "call-get_payment":
                if not json.loads(messages[-1]["content"])["ok"]:
                    repair = calls("get_payment", ORDER)
                    repair["tool_calls"][0]["id"] = "call-retry-payment"
                    return repair
            if "tools" in options and messages[-1].get("tool_call_id") == "call-retry-payment":
                return {"content": self.answer}
            return super().__call__(messages, **options)

    episode = rollout(world, task, execute_reference(world, task),
                      context(world, "recovery"), RepairingTeacher(), student)
    episode.update(id="e", family="f", blueprint_id="b", kind="recovery")
    rows = list(sft_rows(episode))
    assert episode["recovery"]
    assert rows[0]["prompt"][-1]["role"] == "tool"
    assert '"missing_argument"' in rows[0]["prompt"][-1]["content"]
    assert rows[0]["completion"][0]["tool_calls"][0]["id"] == "call-retry-payment"
    assert transcript_key(episode) == transcript_key(copy.deepcopy(episode))


def test_recombination_keeps_read_answers_and_rebinds_actions(world):
    read = Blueprint.model_validate({
        "q": "Ask for the charge.",
        "a_gt": [],
        "o_gt": TASK["o_gt"],
        "evidence": TASK["evidence"],
        "goal_spec": {"success": [], "forbidden": []},
    })
    write = Blueprint.model_validate({**TASK, "q": "Cancel the order with approval.",
                                      "o_gt": [], "evidence": []})

    class Composer(Teacher):
        def __call__(self, messages, **options):
            if messages[0]["content"].startswith("Combine"):
                return {"content": json.dumps({"q": TASK["q"]})}
            return super().__call__(messages, **options)

    merged, verified, reviews = recombine(world, [read, write], context(world, "recombined"),
                                          Composer())
    assert merged.o_gt == TASK["o_gt"]
    assert set(verified["effects"]) == {"orders"}
    assert len(reviews) == 3
    assert verified["reference_trace"][1]["arguments"]["authorization_token"] == "AUTH-001"


def test_amount_can_be_checked_without_the_output_judge(world):
    data = copy.deepcopy(TASK)
    data["evidence"][0]["answer_forms"] = ["$149.00", "14,900 cents"]
    task = Blueprint.model_validate(data)
    with pytest.raises(DataRejected, match="Missing answer"):
        rollout(world, task, execute_reference(world, task), context(world),
                Teacher(answer="You paid $149.001. Do you approve cancellation?"))


def test_ngrok_client_preserves_native_calls_across_gateway_retry(monkeypatch):
    from unittest.mock import Mock

    from apigen_mt_cli import Chat

    result = calls("get_order", ORDER, ("get_payment", ORDER))
    gateway = Mock(status_code=502)
    success = Mock(status_code=200)
    success.json.return_value = {"model": "qwen3.8-flash-next-iq3_s", "usage": {"total_tokens": 10},
                                 "choices": [{"message": result, "finish_reason": "tool_calls"}]}
    post = Mock(side_effect=[gateway, success])
    monkeypatch.setattr("apigen_mt_cli.requests.post", post)
    monkeypatch.setattr("apigen_mt_cli.time.sleep", lambda _: None)
    chat = Chat("https://example.invalid/v1/chat/completions", "strata", "test", strata=True)
    assert chat([{"role": "user", "content": "Look up my order."}], tool_choice="auto") == result
    gateway.json.assert_not_called()
    assert post.call_count == 2
    kwargs = post.call_args.kwargs
    assert kwargs["headers"]["ngrok-skip-browser-warning"] == "true"
    assert kwargs["json"]["reasoning_effort"] == "none"
    assert kwargs["allow_redirects"] is False
    assert chat.tokens["total_tokens"] == 10
