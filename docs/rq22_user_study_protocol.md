# RQ22: User Study Protocol — CollectiveOS Usability Evaluation

## Research Question

**RQ22**: Does CollectiveOS reduce task completion time and perceived cognitive load compared to performing the same tasks manually without AI assistance?

**Hypothesis**: CollectiveOS-assisted task execution will reduce completion time by ≥30% and perceived cognitive load (NASA-TLX score) by ≥20% versus manual baseline.

---

## Study Design

**Type**: Within-subjects, counterbalanced  
**Participants**: N = 12 (target), minimum N = 8 for statistical power (α = 0.05, power = 0.80)  
**Session duration**: 60–75 minutes per participant  
**Compensation**: None required (lab-internal study)

### Conditions

| Condition | Label | Description |
|-----------|-------|-------------|
| A | `manual`    | Participant performs tasks without AI assistance |
| B | `assisted`  | Participant uses CollectiveOS to complete tasks |

Order counterbalanced: half see A→B, half B→A.

---

## Task Set

8 representative tasks spanning three domains:

| ID  | Domain     | Task Description                                                     | Horizon |
|-----|------------|----------------------------------------------------------------------|---------|
| T1  | Navigation | Open Safari, navigate to wikipedia.org, find the article on "LangGraph" | 4–7  |
| T2  | Navigation | Open Finder, navigate to ~/Downloads, sort files by date modified    | 4–7    |
| T3  | Preference  | Open TextEdit, set font to Helvetica 14pt, type a 2-sentence note    | 8–15   |
| T4  | Preference  | Change system volume to 40% and set display brightness to 70%        | 8–15   |
| T5  | Multi-step  | Open Terminal, run `python3 -c "print(2**10)"`, note the output      | 8–15   |
| T6  | Multi-step  | Search for and install a free app from the App Store                 | 16–30  |
| T7  | Memory      | Ask CollectiveOS to remember a preference; verify it is recalled next task | 8–15 |
| T8  | Mixed       | Using Calendar, schedule a meeting for next Monday at 2pm titled "Study debrief" | 16–30 |

T7 is only valid in condition B (assisted). In condition A it is replaced by T2 repeated.

---

## Dependent Variables

### Primary
- **Task completion time (TCT)**: wall-clock seconds from task start to verified completion
- **Task success rate (TSR)**: binary (1 = completed correctly, 0 = gave up or wrong result)

### Secondary
- **NASA-TLX score**: 6-dimension workload rating (Mental, Physical, Temporal, Performance, Effort, Frustration), each 0–100, unweighted mean
- **SUS score**: System Usability Scale — 10-item questionnaire, score 0–100 (condition B only)
- **Error count**: number of wrong actions before success
- **Think-aloud transcript**: qualitative observations coded post hoc

---

## Protocol

### Pre-session (5 min)
1. Brief participant: purpose is to evaluate the system, not the participant.
2. Obtain informed consent (verbal or signed as required).
3. Demographic form: age bracket, years of Mac experience, prior AI assistant use.

### Training (10 min)
- Condition A: show participant the Finder, Spotlight, System Settings.
- Condition B: demonstrate CollectiveOS via voice or text interface with one practice task ("Open Calculator").
- Confirm participant understands the task format.

### Task trials (30–40 min)
For each task:
1. Read task aloud and display on screen.
2. Say: "Please begin when you are ready."
3. Moderator starts timer (stopwatch or `time` logged in session sheet).
4. Participant thinks aloud.
5. On completion (or give-up after 5 minutes): stop timer, note success/failure.
6. 30-second rest between tasks.

### Post-session questionnaires (10 min)
- NASA-TLX (after each condition, or once at end for both — mark on sheet).
- SUS questionnaire (condition B only).
- Open-ended feedback (2–3 minutes): "What worked well? What was frustrating?"

---

## Data Collection Sheet

```
Participant ID: ___________    Date: ___________    Order: A→B / B→A
Condition: manual / assisted

Task | Start time | End time | TCT (s) | Success (Y/N) | Errors | Notes
-----|------------|----------|---------|---------------|--------|------
T1   |            |          |         |               |        |
T2   |            |          |         |               |        |
T3   |            |          |         |               |        |
T4   |            |          |         |               |        |
T5   |            |          |         |               |        |
T6   |            |          |         |               |        |
T7   |            |          |         |               |        |
T8   |            |          |         |               |        |

NASA-TLX (0–100 each):
  Mental Demand: ___  Physical Demand: ___  Temporal Demand: ___
  Performance:   ___  Effort:           ___  Frustration:     ___
  Mean TLX: ___

SUS (condition B only, 1–5 scale, sum × 2.5):
  Q1 __ Q2 __ Q3 __ Q4 __ Q5 __ Q6 __ Q7 __ Q8 __ Q9 __ Q10 __
  SUS Score: ___
```

---

## Analysis Plan

### Quantitative
- **Paired t-test** (or Wilcoxon signed-rank if non-normal): TCT manual vs. assisted per task
- **Effect size**: Cohen's d
- **Success rate**: McNemar's test for paired binary outcomes
- **NASA-TLX**: paired t-test on mean TLX score
- Report mean ± SD for each measure; p < 0.05 considered significant

### Qualitative
- Code think-aloud transcripts for: confusion events, recovery strategies, trust signals, frustration cues
- Affinity diagram across participants

---

## Ethical Considerations

- No personally identifiable information stored with task data.
- Participants may withdraw at any time without consequence.
- Screen recordings (if used) stored locally, deleted after analysis.
- No deception used; purpose disclosed upfront.

---

## Recruitment Criteria

- Age 18+, fluent in English
- Regular macOS user (≥ 1 year experience)
- No prior exposure to CollectiveOS
- Not a member of the development team

---

## Metrics Summary for RQ22

| Metric            | Formula                              | Target           |
|-------------------|--------------------------------------|------------------|
| TCT reduction     | (TCT_manual − TCT_assisted) / TCT_manual | ≥ 30%        |
| TLX reduction     | (TLX_manual − TLX_assisted) / TLX_manual | ≥ 20%        |
| TSR improvement   | TSR_assisted − TSR_manual            | ≥ 0.10          |
| SUS score         | Mean SUS (condition B)               | ≥ 68 (acceptable) |

---

*This document is the study protocol for RQ22. Run participants only after IRB/ethics review or team sign-off. Raw data files go in `docs/study_data/` (git-ignored).*
