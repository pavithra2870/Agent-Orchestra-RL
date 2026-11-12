# Result tables

Evaluation: 1410 scenarios over 7 test sets, 4 runs each; learning agents trained with seeds [0, 1, 2] (mean ± std over seeds).

## 1. Overall (all test sets pooled)

| Agent | Success | pass^4 | Rule breaks | Regret | Steps | Cost | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based playbook | 99.9% | 99.6% | 0.0% | 0.05 | 5.3 | 5.0 | 6.3 |
| Random (legal actions) | 7.6% | 0.4% | 0.0% | 7.78 | 2.1 | 4.6 | 5.7 |
| DQN | 98.5% ±0.4 | 98.1% ±0.3 | 0.0% | 0.23 ±0.03 | 5.9 ±0.1 | 5.4 ±0.1 | 6.9 ±0.1 |
| PPO | 94.0% ±0.7 | 93.6% ±0.7 | 0.0% | 0.84 ±0.06 | 7.2 ±0.4 | 6.6 ±0.3 | 8.7 ±0.4 |
| CQL (offline) | 97.6% ±0.9 | 97.3% ±0.9 | 0.0% | 0.24 ±0.07 | 5.3 ±0.0 | 5.0 ±0.0 | 6.4 ±0.0 |
| Q-learning | 68.0% ±0.3 | 55.8% ±0.1 | 0.0% | 2.84 ±0.03 | 4.5 ±0.1 | 5.1 ±0.0 | 6.4 ±0.1 |
| SARSA | 66.9% ±1.3 | 54.3% ±2.3 | 0.0% | 2.93 ±0.12 | 4.5 ±0.1 | 5.1 ±0.1 | 6.4 ±0.1 |
| DQN, trained with tool-failure scenarios | 99.6% ±0.1 | 99.0% ±0.2 | 0.0% | 0.24 ±0.02 | 6.3 ±0.1 | 5.8 ±0.1 | 7.5 ±0.1 |
| PPO, trained with tool-failure scenarios | 96.6% ±2.0 | 96.1% ±2.0 | 0.0% | 0.72 ±0.16 | 7.7 ±0.2 | 7.1 ±0.1 | 9.2 ±0.3 |
| DQN, outcome-only reward | 98.7% ±0.2 | 98.4% ±0.1 | 0.0% | 0.21 ±0.02 | 5.9 ±0.0 | 5.3 ±0.0 | 6.9 ±0.1 |
| PPO, outcome-only reward | 96.3% ±1.6 | 95.6% ±1.8 | 0.0% | 0.73 ±0.06 | 7.8 ±0.5 | 7.0 ±0.5 | 9.2 ±0.5 |
| CQL without conservative penalty | 98.7% ±0.1 | 98.3% ±0.2 | 0.0% | 0.53 ±0.06 | 7.6 ±0.2 | 6.7 ±0.2 | 9.1 ±0.3 |
| DQN, no guardrail | 21.1% ±5.1 | 19.2% ±4.7 | 16.6% ±14.8 | 9.51 ±4.03 | 9.8 ±4.6 | 6.4 ±2.9 | 8.3 ±4.1 |
| PPO, no guardrail | 7.8% ±2.0 | 7.5% ±2.2 | 0.0% | 7.37 ±0.55 | 2.6 ±2.0 | 3.5 ±3.3 | 3.5 ±3.3 |
| Random, no guardrail | 1.0% | 0.0% | 83.2% | 35.25 | 7.2 | 11.1 | 12.5 |

## 2. Success rate by test set

| Agent | iid | heldout_combos | tool_failures | missing_info | out_of_scope | returns | policy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based playbook | 100.0% | 100.0% | 99.1% | 99.8% | 100.0% | 100.0% | 100.0% |
| Random (legal actions) | 6.0% | 3.5% | 6.4% | 4.3% | 49.6% | 5.8% | 7.5% |
| DQN | 100.0% | 99.6% ±0.1 | 91.6% ±0.2 | 99.9% ±0.1 | 100.0% | 98.8% ±1.7 | 100.0% |
| PPO | 99.9% ±0.2 | 66.6% ±5.0 | 91.4% ±1.0 | 99.8% | 100.0% | 100.0% | 99.7% ±0.2 |
| CQL (offline) | 99.1% ±0.6 | 97.2% ±2.1 | 93.9% ±0.9 | 97.2% ±1.9 | 100.0% | 96.8% ±4.6 | 100.0% |
| Q-learning | 92.5% ±0.1 | 6.8% ±0.1 | 79.0% ±1.5 | 87.4% ±3.2 | 100.0% | 60.1% ±0.6 | 69.0% ±0.7 |
| SARSA | 91.3% ±1.1 | 7.0% ±0.2 | 77.7% ±1.1 | 86.6% ±3.4 | 100.0% | 56.8% ±2.7 | 69.6% ±1.5 |
| DQN, trained with tool-failure scenarios | 99.9% ±0.1 | 99.7% ±0.2 | 98.2% ±0.8 | 99.4% ±0.3 | 100.0% | 100.0% | 99.7% ±0.5 |
| PPO, trained with tool-failure scenarios | 100.0% | 77.6% ±14.1 | 98.5% ±0.3 | 99.8% | 100.0% | 100.0% | 100.0% ±0.1 |
| DQN, outcome-only reward | 100.0% | 100.0% | 91.3% ±1.6 | 99.8% ±0.1 | 100.0% | 100.0% | 100.0% |
| PPO, outcome-only reward | 100.0% | 82.0% ±12.9 | 92.7% ±2.0 | 99.8% ±0.1 | 100.0% | 99.5% ±0.7 | 100.0% |
| CQL without conservative penalty | 100.0% | 99.7% ±0.2 | 91.8% ±1.3 | 99.6% ±0.6 | 100.0% | 100.0% | 100.0% |
| DQN, no guardrail | 21.5% ±5.3 | 10.6% ±6.5 | 14.3% ±6.6 | 22.2% ±7.3 | 100.0% | 2.5% ±3.3 | 41.1% ±13.4 |
| PPO, no guardrail | 6.4% ±1.4 | 2.0% ±2.8 | 5.0% ±2.9 | 7.5% ±4.0 | 33.3% ±47.1 | 2.9% ±4.1 | 18.6% ±13.7 |
| Random, no guardrail | 0.6% | 0.4% | 0.9% | 0.3% | 9.6% | 0.9% | 0.8% |

## 3. Asking and handing over (overall)

| Agent | Asks when needed | Asks only when needed | Hands over when needed | Hands over only when needed |
|---|---:|---:|---:|---:|
| Rule-based playbook | 81.6% | 81.6% | 100.0% | 99.4% |
| Random (legal actions) | 27.7% | 46.6% | 50.4% | 18.0% |
| DQN | 80.9% ±0.9 | 74.4% ±0.6 | 94.5% ±0.2 | 98.5% ±1.6 |
| PPO | 80.9% ±0.4 | 71.5% ±0.6 | 91.3% ±0.8 | 97.7% ±1.3 |
| CQL (offline) | 80.6% ±0.7 | 75.6% ±0.4 | 96.0% ±0.6 | 94.9% ±3.3 |
| Q-learning | 65.4% ±1.1 | 62.1% ±1.5 | 84.0% ±0.1 | 50.6% ±0.4 |
| SARSA | 64.0% ±1.1 | 62.1% ±0.9 | 83.0% ±1.4 | 48.2% ±1.3 |

## 4. Sample efficiency (validation split)

| Method | Budget | To reach 90% | Env steps to 90% | Best validation | Final validation | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| Q-learning | 5,000 episodes | 4000 (3/3 seeds) | 15534 | 94.0% ±0.5 | 93.8% ±0.3 | 21 |
| SARSA | 5,000 episodes | 4400 (3/3 seeds) | 17314 | 93.8% ±2.2 | 93.8% ±2.2 | 14 |
| DQN | 4,000 episodes | 933 (3/3 seeds) | 2674 | 100.0% | 100.0% | 53 |
| PPO | 12,000 episodes | 4600 (3/3 seeds) | 19574 | 100.0% | 99.8% ±0.3 | 84 |
| CQL (offline) | 20,000 gradient steps | 1000 (3/3 seeds) | 0 | 100.0% | 100.0% | 45 |
| DQN, outcome-only reward | 4,000 episodes | 867 (3/3 seeds) | 2432 | 100.0% | 100.0% | 48 |
| PPO, outcome-only reward | 12,000 episodes | 6867 (3/3 seeds) | 24100 | 100.0% | 100.0% | 74 |
| DQN, trained with tool-failure scenarios | 4,000 episodes | 800 (3/3 seeds) | 2237 | 100.0% | 99.3% ±0.5 | 61 |
| PPO, trained with tool-failure scenarios | 12,000 episodes | 4600 (3/3 seeds) | 19778 | 100.0% | 99.8% ±0.3 | 93 |
| DQN, no guardrail | 4,000 episodes | never (0/3 seeds) | - | 26.0% ±6.8 | 24.9% ±8.2 | 75 |
| PPO, no guardrail | 12,000 episodes | never (0/3 seeds) | - | 7.1% ±1.4 | 1.8% ±2.5 | 35 |
| CQL without conservative penalty | 20,000 gradient steps | 1000 (3/3 seeds) | 0 | 100.0% | 98.4% ±0.3 | 49 |

## 5. Router on three wording sets

| Messages | Goal set exactly right | Out-of-scope caught | Mean similarity |
|---|---:|---:|---:|
| Seen wording (set A) | 97.2% | 92.1% | 0.89 |
| Tuning wording (set C) | 84.8% | 71.0% | 0.76 |
| Held-out wording (set B), final test | 93.8% | 100.0% | 0.79 |
| Held-out (B) with more examples (A+C) | 92.0% | 100.0% | 0.81 |

### LLM parser (phi4-mini:latest, 60 held-out messages)

| Method | Exactly right | Invented a goal | Missed a goal | Low confidence on errors |
|---|---:|---:|---:|---:|
| Router only | 91.7% | 8.3% | 8.3% | 0% |
| LLM only (raw) | 81.7% | 13.3% | 18.3% | - |
| LLM + router cross-check | 81.7% | 13.3% | 18.3% | 91% |

Average 4.9 s per message; 0 goal(s) dropped by the guard.

## 6. End to end with the real router parser (200 held-out messages, goals read exactly right in 93.0%)

| Agent | Oracle parser | Router parser | Cost of language errors |
|---|---:|---:|---:|
| rule_based | 100.0% | 95.0% | 5.0 pts |
| q_learning | 63.2% ±1.3 | 59.7% ±1.7 | 3.6 pts |
| sarsa | 62.8% ±1.9 | 63.5% ±1.3 | -0.7 pts |
| dqn | 99.8% ±0.2 | 94.9% ±0.3 | 4.8 pts |
| ppo | 87.0% ±1.4 | 78.7% ±1.0 | 8.3 pts |
| cql | 99.0% ±0.7 | 94.2% ±0.6 | 4.8 pts |

Reply checker: 0/120 template replies flagged; LLM drafts rejected 8/25; unsupported claims in shipped replies 0/25; 5.1 s per LLM reply.

## 7. LLM choosing the tools (phi4-mini:latest, 60 scenarios, one run each)

| Agent | Success | Rule breaks | Blocked attempts/run | Regret | Seconds per conversation |
|---|---:|---:|---:|---:|---:|
| LLM | 28.3% | 0.0% | 0.00 | 6.69 | 40.5 |
| rule_based | 100.0% | 0.0% | 0.00 | 0.03 | <0.1 |
| random | 16.7% | 0.0% | 0.00 | 7.00 | <0.1 |
| dqn | 100.0% | 0.0% | 0.00 | 0.07 | <0.1 |
| ppo | 91.7% | 0.0% | 0.00 | 0.91 | <0.1 |
| cql | 98.3% | 0.0% | 0.00 | 0.16 | <0.1 |
| q_learning | 78.3% | 0.0% | 0.00 | 1.99 | <0.1 |
| sarsa | 78.3% | 0.0% | 0.00 | 1.94 | <0.1 |

