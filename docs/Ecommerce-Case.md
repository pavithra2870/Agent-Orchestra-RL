# Report: reinforcement learning for e-commerce customer support

All numbers below are measured results, except where a sentence says otherwise.
The shop, customers and tools are simulated. The original project was not changed.
The results from before the returns / damage / policy extension ("Group B") measure a different, smaller task and are not comparable with the numbers here.

## 1. Summary

An RL policy decides, step by step, which tool a support agent should call (or whether to ask the customer or hand over to a human).
A language layer turns the message into goals and words the reply; the policy never reads text. Seven decision-makers were compared on
1,410 held-out scenarios over 7 test sets (4 runs each; learning agents trained with 3 seeds), plus ablations and a local LLM benchmark.
The agent handles six goals: delivery status, payment change, refund, cancel, **return / exchange / damage claim** and **policy questions**.

| What we found | Number |
|---|---|
| Best learning agent overall | **DQN, 98.5%** success (pass^4 98.1%, regret 0.23); with failure-aware training **99.6%** |
| Offline RL (no live training episodes) | **CQL, 97.6%** success, learned only from a log of 4,000 past conversations |
| Hand-written rule-based playbook | **99.9%**: the RL agents did **not** beat it on success rate |
| Random agent (only legal actions) | 7.6% |
| Safety rule violations (all agents with the guardrail) | **0.0%**, including 0 refunds before an inspection passed |
| Without the guardrail | DQN 21.1% success with 16.6% rule breaks; a random agent breaks a rule in **83.2%** of runs |
| Unseen goal combinations | DQN 99.6%, CQL 97.2%, **PPO 66.6%**, Q-learning 6.8%, SARSA 7.0% |
| Returns and damage claims (new) | DQN 98.8%, PPO 100%, CQL 96.8%, Q-learning 60.1%, SARSA 56.8% |
| Policy questions (new) | DQN 100%, CQL 100%, PPO 99.7%, Q-learning 69.0%, SARSA 69.6% |
| Language layer on unseen wording | router reads the goals exactly right in **93.8%** (92.0% with more examples) |
| End to end with the real router | agents lose **up to 8.3 points** compared with being handed the true goals (DQN 4.8, CQL 4.8, PPO 8.3) |

**Which agent is best?** The rule-based playbook scores highest, because the shop's rules are known and a person wrote them down. Among *learned* policies DQN is
best and fastest (90% validation success after 933 episodes, about 2,700 environment steps). CQL is the best choice when only historical logs exist.
**Which metrics matter most?** Success (judged from the database), pass^4 (consistency), safety violations (must be 0), regret against the oracle, and
success on held-out and stressed test sets, because the overall number hides large differences (section 4.2).

## 2. Scope: what this is and is not

- **Is:** an RL-controlled support agent for a *simulated* shop, engineered to production-grade quality: typed and versioned configuration, seeded and reproducible runs,
  hard safety rules in code, an independent evaluator, held-out tests, an automated test suite (216 tests), run tracking, background jobs, an API and a dashboard.
- **Is not:** a deployment. It never touches a real customer, payment or refund system. Real traffic would also need privacy handling, monitoring and integration work.

## 3. Method

### 3.1 The problem as a Markov decision process

| Element | Definition |
|---|---|
| State (85 numbers) | which goals were found / handled / need a human; whether the customer and order are verified; what each tool revealed (shipment status, eligibility results); whether the customer still has to name a payment method or answer the return questions; failures per tool; steps and cost used; parser confidence and scope score; the last action. It contains **no hidden ground truth**. |
| Actions (16) | 13 tools (`lookup_customer`, `lookup_order`, `check_shipment`, `get_payment_details`, `check_payment_change_eligibility`, `update_payment_method`, `check_refund_eligibility`, `initiate_refund`, `check_cancel_eligibility`, `cancel_order`, `check_return_eligibility`, `process_return`, `lookup_policy`), `request_clarification`, `escalate_to_human`, `finish_task`. The policy picks the tool; the environment fills in ids, the customer's answers and the policy topic from verified context. |
| Reward | small cost and latency penalties per step; +1 when a requested goal is handled; at the end +5 for a fully correct run or -2 otherwise (decided by the independent evaluator); -5 per executed safety violation; small penalties for blocked and unneeded actions. All weights are set in a YAML config. |
| Episode ends | `finish_task`, `escalate_to_human`, or 16 steps. |

Rules of the shop (all numbers are set in the config):
- **Payment** can change only on unpaid open orders, and only to a method the customer named (the agent may never pick one).
- **Refund** depends on status and a high-value limit (large refunds need a human). A *delivered* item is never refunded directly: the customer returns it and the refund follows once it passes inspection.
- **Cancel** only before shipping.
- **Damaged item:** reported within 48 hours of delivery; the customer chooses exchange or return-and-refund and confirms the condition. Orders up to 500 rupees with fewer than 2 damage claims in 30 days (and one claim per order, courier-confirmed delivery, paid) are refunded straight away or an exchange is opened; everything else goes to staff.
- **Change of mind:** 7 days, no fee, the customer confirms the item is unused; a human supply-chain team inspects it (simulated). Passed: refund or replacement. Failed: nothing is refunded and the item goes back.
- **Policy questions** are answered by the `lookup_policy` tool from texts generated from the same numbers; they need no order and change nothing.
- Every refund, cancel, payment change and return asks the customer yes or no first (a step before the agent runs).

The shop has six goals, multi-goal requests, five scenario types (standard, missing order id, wrong order id, low parser confidence, out of scope), random tool failures and sometimes a tool that is down.

### 3.2 Guardrails (four layers)

1. **Input:** embedding router with a scope score and a calibrated confidence; typo repair from the router's own example words.
2. **Actions (hard rules in code):** a state-changing tool is legal only with a verified order, a passed eligibility check, a matching customer request and, for payments and returns, the answers only the customer can give. Retries are capped, then a human takes over. Illegal actions are masked out of the policy *and* out of the learning targets.
3. **Output:** the reply is built from verified tool results. An optional LLM rewording is fact-checked and replaced by the template if anything is unsupported; questions and policy answers are never LLM-worded.
4. **Escalation:** out of scope, large refund, damage claims that need a review, or repeated tool failure go to a human.

### 3.3 Evaluation design

- **Independent evaluator:** success is decided from the scenario's ground truth, the final database state and the trace, never from the agent's own words. A refund of a delivered item without an instant-refund claim or a passed inspection is a violation.
- **Test sets:** `iid`, `heldout_combos` (five goal combinations never seen in training), `tool_failures`, `missing_info`, `out_of_scope`, **`returns`** (all twelve return outcomes) and **`policy`**; `paraphrase` (unseen wording) for the real-parser pipeline. Each set has its own seed range and a test checks that none share a scenario.
- **Repeats and seeds:** 4 runs per scenario (tool failures differ), 3 training seeds per learning agent.
- **Metrics:** success, pass^4, safety violations, regret vs an oracle that knows the hidden truth (tool failures off), steps, cost, latency, asking and hand-over quality.
- **Language layer protocol:** three wording sets: A = router examples, C = tuning and calibration only, B = final test only. Set B was replaced by a fresh set in Group B because the first one had been read while improving the examples.

## 4. Results

### 4.1 Overall ranking (all test sets pooled)

| Agent | Success | pass^4 | Rule breaks | Regret | Steps |
|---|---:|---:|---:|---:|---:|
| Rule-based playbook | 99.9% | 99.6% | 0.0% | 0.05 | 5.3 |
| DQN, trained with tool-failure scenarios | 99.6% ±0.1 | 99.0% ±0.2 | 0.0% | 0.24 ±0.02 | 6.3 |
| DQN | 98.5% ±0.4 | 98.1% ±0.3 | 0.0% | 0.23 ±0.03 | 5.9 |
| CQL (offline) | 97.6% ±0.9 | 97.3% ±0.9 | 0.0% | 0.24 ±0.07 | 5.3 |
| PPO, trained with tool-failure scenarios | 96.6% ±2.0 | 96.1% ±2.0 | 0.0% | 0.72 ±0.16 | 7.7 |
| PPO | 94.0% ±0.7 | 93.6% ±0.7 | 0.0% | 0.84 ±0.06 | 7.2 |
| Q-learning | 68.0% ±0.3 | 55.8% ±0.1 | 0.0% | 2.84 ±0.03 | 4.5 |
| SARSA | 66.9% ±1.3 | 54.3% ±2.3 | 0.0% | 2.93 ±0.12 | 4.5 |
| Random (legal actions) | 7.6% | 0.4% | 0.0% | 7.78 | 2.1 |

(The oracle reaches 100% on every test set when tool failures are switched off, which is how regret is defined; the rule-based playbook does too.)

### 4.2 The overall number hides the real differences

| Agent | iid | held-out combos | tool failures | missing info | out of scope | returns | policy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based playbook | 100.0% | 100.0% | 99.1% | 99.8% | 100.0% | 100.0% | 100.0% |
| DQN | 100.0% | 99.6% ±0.1 | 91.6% ±0.2 | 99.9% ±0.1 | 100.0% | 98.8% ±1.7 | 100.0% |
| CQL (offline) | 99.1% ±0.6 | 97.2% ±2.1 | 93.9% ±0.9 | 97.2% ±1.9 | 100.0% | 96.8% ±4.6 | 100.0% |
| PPO | 99.9% ±0.2 | **66.6% ±5.0** | 91.4% ±1.0 | 99.8% | 100.0% | 100.0% | 99.7% ±0.2 |
| Q-learning | 92.5% ±0.1 | **6.8% ±0.1** | 79.0% ±1.5 | 87.4% ±3.2 | 100.0% | 60.1% ±0.6 | 69.0% ±0.7 |
| SARSA | 91.3% ±1.1 | **7.0% ±0.2** | 77.7% ±1.1 | 86.6% ±3.4 | 100.0% | 56.8% ±2.7 | 69.6% ±1.5 |

On familiar scenarios almost every method looks fine. **On goal combinations never seen in training, PPO and both tabular methods fail, while DQN and CQL do not.**
Broken down by combination, PPO is near-perfect on every trained combination and on three of the five held-out ones (delivery+cancel, delivery+return, policy+delivery: 100%),
but fails the two that combine several of the older goals (payment+refund 25%, delivery+payment+refund 6%). Why DQN and CQL generalise and PPO does not is **not tested**.
The tabular methods (Q-learning, SARSA) also struggle on the new goals (returns about 57-60%, policy about 69%): their compressed state has to be re-learned as the task grows.

### 4.3 Returns and damage claims, by outcome (returns test set)

| Outcome the agent has to produce | DQN | PPO | CQL | Rule-based |
|---|---:|---:|---:|---:|
| Refund a small damaged order right away | 96% | 100% | 100% | 100% |
| Open an exchange | 100% | 100% | 100% | 100% |
| Open a return (change of mind) | 98% | 100% | 100% | 100% |
| Hand to staff: over the 500-rupee cap | 100% | 100% | 85% | 100% |
| Hand to staff: repeat claimer | 100% | 100% | 80% | 100% |
| Hand to staff: late report (over 48 hours) | 100% | 100% | 94% | 100% |
| Hand to staff: high-value change of mind | 100% | 100% | 86% | 100% |
| Decline (not delivered / unpaid / already claimed / window passed / condition not confirmed) | 97-100% | 100% | 100% | 100% |

The offline agent (CQL) is the weakest at knowing when a human must decide: its log came from a mostly sensible agent, and the rarer hand-over cases are thin in it.
No agent with the guardrail refunded anything before an inspection passed (the evaluator would count it as a violation).

### 4.4 Learning speed (validation split, steps to reach 90%)

| Method | To 90% | Environment steps | Budget |
|---|---:|---:|---|
| DQN | 933 episodes | 2,674 | 4,000 episodes |
| CQL | 1,000 gradient steps | **0** live steps | 20,000 gradient steps |
| Q-learning | 4,000 episodes | 15,534 | 5,000 episodes |
| SARSA | 4,400 episodes | 17,314 | 5,000 episodes |
| PPO | 4,600 episodes | 19,574 | 12,000 episodes |

DQN needs about 4 times fewer episodes than Q-learning. PPO's tuning no longer separates its settings (all five reach 100% validation by episode 1,800), so the library defaults were used.

### 4.5 Offline RL

CQL never interacts with the simulator while learning. It trains on a log of 4,000 conversations made by a rule-based agent that sometimes moves randomly plus some fully random runs,
and reaches 97.6%. Removing its conservative penalty gives a *higher* success rate (98.7%) but a wasteful agent (7.6 steps vs 5.3, regret 0.53 vs 0.24), so the penalty buys efficiency here rather than accuracy.
The trainer refuses logs under 4,000 episodes (an earlier leftover small log once produced misleading CQL results) and records the dataset in each run.

### 4.6 What each design choice buys

| Change | Before | After |
|---|---:|---:|
| No guardrail, DQN | 98.5% | 21.1% (16.6% rule breaks) |
| No guardrail, PPO | 94.0% | 7.8% |
| No guardrail, random agent | rule breaks 0.0% | rule breaks **83.2%** |
| Outcome-only reward, DQN | 98.5% | 98.7% |
| Outcome-only reward, PPO | 94.0% | 96.3% ±1.6 |
| Train with tool-failure scenarios, DQN, tool-failure test set | 91.6% | **98.2%** |
| Train with tool-failure scenarios, PPO, tool-failure test set | 91.4% | **98.5%** |

- **The guardrail** makes unsafe behaviour impossible for any policy, and also makes the task learnable: unguarded learners fail to learn it.
- **Reward shape:** removing the +1 per handled goal changed nothing material, so the study does not show that dense rewards are needed here.
- **Tool failures:** every plain learner loses 5 to 9 points under tool failures, finishing instead of handing over when a tool stays down. Training with such scenarios (30% of training) removes the gap for DQN and PPO.

### 4.7 Language layer

| Messages | Goal set exactly right | Out-of-scope caught |
|---|---:|---:|
| Seen wording (A) | 97.2% | 92.1% |
| Tuning wording (C) | 84.8% | 71.0% |
| **Held-out wording (B, fresh set), final test** | **93.8%** | 100.0% |
| Held-out (B) with more example sentences (A+C) | 92.0% | 100.0% |

- Adding the two new classes first dropped the held-out accuracy from 90.8% to 79.6%; broader examples and typo repair brought it back. The old and the new test set differ, so 90.8% and 93.8% are not directly comparable.
- **Policy questions** (21 held-out): the router recognises 95.2% and finds the right topic for 95.2% (both right 90.5%); 1.2% of ordinary requests are wrongly called policy questions.
- The router's confidence flagged **none** of its wrong parses on the 200-message end-to-end sample, so by itself it is not a usable error detector.
- On 60 held-out messages: router only 91.7% exact, raw LLM 81.7% (invents a goal in 13.3%, misses 18.3%), LLM + router cross-check 81.7% with **90.9%** of its errors flagged as low confidence. The LLM takes 4.9 s per message. The cross-check is a safety signal, not an accuracy gain, and the sample is small. The LLM read "i want to return my order" as a refund; a rule now treats a return as a return unless the customer asks for money back.
- **End to end with the real router (200 messages):** the goal set is read exactly right in 93.0%. The agents lose up to 8.3 points compared with being handed the true goals: rule-based 100% to 95.0%, DQN 99.8% to 94.9%, CQL 99.0% to 94.2%, PPO 87.0% to 78.7%, Q-learning 63.2% to 59.7%, SARSA 62.8% to 63.5% (no loss, within noise). One cause of the loss was found late and fixed: the real router is very confident (about 0.94) that an off-topic message is off-topic, a state the simulator never produced, so the learned agents wandered instead of handing it to a human; before the fix the same measurement gave 3.3 to 9.8 points (DQN 93.4%, CQL 90.6%).
- **Replies:** 0 of 120 template replies flagged by the checker. Of 25 LLM drafts, **8 were rejected** and replaced by the verified template, and 0 unsupported claims were shipped (5.1 s per LLM reply). The checker was tightened while taking demo screenshots: the LLM had written "$180" in a rupee shop and "ready for dispatch", neither of which was a fact.

### 4.8 An LLM choosing the tools

phi4-mini (3.8B parameters, on this laptop's CPU), shown the same state and the same legal actions as the RL agents, re-measured on the Group B task (60 scenarios: 30 iid, 9 held-out combinations, 12 missing-information, 9 out of scope; one run each): **28.3%** success, **40.5 s** per conversation. The RL agents on the same sample: DQN 100%, CQL 98.3%, PPO 91.7%, rule-based 100%, Q-learning and SARSA 78.3%, random 16.7%. Its most common failure is handing over to a human when it should not (32 of the 43 failed runs); it also missed 11 required hand-overs. This says a small local LLM is a poor decision-maker for this task. It does not say that a much larger LLM would be; that was not tested. (Before Group B the same benchmark gave 28.3% and 38.6 s on the older task.)

## 5. What worked and what did not

**Worked**
- The guardrail design: zero rule breaks for every guarded agent, and it makes learning feasible. New safety rules (the agent may not choose a payment method, may not open a return before the customer has answered) were added as masks.
- DQN as the all-round learned policy; CQL as the offline option; failure-aware training for robustness.
- The evaluation framework: it exposed generalisation and robustness gaps that the headline number hid, and its own tests caught bugs in the environment and the reply layer.
- The wording-set protocol, and replacing the test set when it had been read.
- Building the new skills as part of the RL environment (two new tools plus a policy tool) rather than as shortcuts: the same agents learned them with the same budgets.

**Did not work / surprised us**
- No learned agent beat the hand-written playbook on success rate. In this simulator the rules are fully known, so the playbook is already near-perfect.
- PPO still fails some unseen goal combinations (66.6%), and the tabular methods fail almost all of them.
- The offline agent is the weakest at deciding when a human must take over.
- The router's confidence score is not usable as an error detector; language errors cost the agents 3 to 10 points end to end.
- LLM wording and parsing remain the least reliable parts: 8 of 25 reply drafts were rejected, the LLM misread returns as refunds, and it once invented a currency and a delivery status.

**Mistakes made and fixed**: an environment flaw found by the oracle test; a scenario-generator bug; shell escaping that silently corrupted regexes (twice more in Group B); reply fact-checker bugs; a PPO budget mistaken for a weak algorithm; a tuning protocol that leaked; CQL trained on a leftover log; the agent changing a payment method nobody named; a demo database that reset on every message; a router that lost 11 points when two classes were added; and a test set that stopped being a test once I read its errors.

## 6. Limitations

- Everything is simulated. The shop's rules are simple and fully known, which is why a hand-written playbook can be perfect. The value of RL would be larger where rules are hard to write down.
- Three training seeds per method; some spreads are large (PPO with failure training: 77.6% ±14.1 on held-out combinations).
- Only five goal combinations are held out; the explanation for PPO's failure is untested.
- The environment's terminal reward uses the same evaluator code as the final check. The evaluator ignores the agent's claims, but the agents are optimised for what it measures.
- A customer lying about the condition of an item is not part of what the agent has to detect: the inspection outcome is simulated after the episode.
- The agent does not check a policy before acting on a request; it looks a policy up only when the customer asks.
- The LLM results use a 3.8B model on CPU and small samples.
- The clause splitter's connector words were written knowing the wording in sets B and C, so the held-out test measures goal wording, not connector wording.
- No user study, no real customers, no latency under load.

## 7. Suggested next steps

1. Test why PPO fails on unseen combinations (for example by adding goal-permutation augmentation or training on more combinations).
2. Make the router's uncertainty more informative, or use disagreement with the LLM to trigger the clarification question.
3. Let the agent check a policy before it acts (for example a refund request that conflicts with a stated policy).
4. Add more hand-over cases to the offline log so CQL learns when a human must decide.
5. Repeat the LLM benchmark with a larger model.

## 8. How to reproduce

```powershell
# tests
.\venv\Scripts\python.exe -m pytest ecommerce/tests -q
# official training (3 seeds), tuning, and evaluation
.\venv\Scripts\python.exe -m ecommerce.training.run_training --suite full --seeds 0 1 2
.\venv\Scripts\python.exe -m ecommerce.training.run_training --agents dqn ppo --variant robust --seeds 0 1 2
.\venv\Scripts\python.exe -m ecommerce.training.tune_ppo
.\venv\Scripts\python.exe -m ecommerce.evaluation.run_eval
.\venv\Scripts\python.exe -m ecommerce.evaluation.run_pipeline_eval --llm-replies 25
.\venv\Scripts\python.exe -m ecommerce.language.evaluate_language --llm 60
.\venv\Scripts\python.exe -m ecommerce.language.evaluate_knowledge
# dashboard (http://127.0.0.1:8000)
.\venv\Scripts\python.exe -m ecommerce.backend.main
```

Note: CQL runs need no extra step (the 4,000-episode log is created on first use).
