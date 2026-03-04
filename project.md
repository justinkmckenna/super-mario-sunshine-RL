# Super Mario Sunshine RL Project

## Goal

Train an RL agent to complete a constrained task in Super Mario Sunshine, starting with the Blooper Surfing Safari level.

This project will start with a narrow environment definition and a fixed episode start rather than trying to train an agent to play the full game.

## First Target

Mission: Blooper Surfing Safari

Why this mission:

- It is much narrower than general 3D platforming.
- It resembles a race or track-following task.
- Success criteria are clear: stay on course, avoid obstacles, finish.
- It should allow a limited action space compared to normal Sunshine gameplay.

## Core Assumptions

- Emulator: Dolphin on Windows
- Training and development machine: Windows laptop with RTX 4050
- Platform support target for v0: Windows only
- Control/automation: begin with the simplest reliable path, but do not assume keyboard input is sufficient long-term
- First version observations: raw pixels
- First version environment logic: memory reads are allowed for reward, done, and reset signals
- First version scope: one fixed mission from one fixed savestate
- Initial implementation goal: robust environment loop plus random/scripted baselines before RL training

## Important Constraints

- Raw pixels are acceptable for policy input in v1, but reward, done, and reset logic may still need game-aware signals.
- Full-game Sunshine is out of scope for the initial version.
- Fast and deterministic reset matters as much as model quality.
- Action space should stay as small as possible at the start.
- The environment API should be Gymnasium-compatible from the start.

## Environment Contract V0

### Observation

First-pass idea:

- Raw emulator frames
- Cropped to the gameplay region if needed
- Downsampled to a smaller resolution
- Possibly grayscale
- Possibly stacked across several frames for motion information

Open questions:

- Final observation size
- Whether grayscale is sufficient
- Whether frame stacking is enough to capture movement and heading changes

### Action Space

First-pass goal: keep it narrow.

Candidate action space:

- steer left
- steer neutral
- steer right

Open questions:

- Does Blooper Surfing require analog steering magnitude, or will coarse discretization work?
- Are additional actions needed beyond steering for starting, recovery, or menu flow?
- Is forward movement effectively automatic during the mission?
- What is the most reliable control injection path into Dolphin on Windows?

### Reward

Initial reward design candidate:

- positive reward for forward progress along the course
- large positive reward for completing the course
- large negative reward for falling or failing
- small per-step penalty to encourage faster completion

Risks:

- Rewarding generic movement instead of true course progress
- Reward shaping that encourages oscillation or unsafe shortcuts
- Sparse reward if progress is not measured correctly
- Purely visual reward logic may be slower and less reliable than using memory reads

### Done Conditions

Episode should end on:

- mission success / finish
- fall off course or equivalent failure
- timeout / max steps
- unrecoverable mission reset state

### Reset

Preferred approach:

- load a savestate positioned immediately before the playable part of the mission

Requirements:

- reset must be consistent
- reset must be fast enough for training
- episode should start from the same initial state in v1
- some small start-state nondeterminism should be expected and measured rather than assumed away

## Milestones

### 1. Emulator Bring-Up

Goal:

- Launch Dolphin reliably
- Boot Super Mario Sunshine
- Create and restore a Blooper Surfing Safari savestate

Exit criteria:

- One repeatable process starts the mission from a fixed point

### 2. Input Automation

Goal:

- Send programmatic controller inputs to Dolphin
- Verify left, right, and neutral behavior

Exit criteria:

- A script can make the blooper visibly respond to controlled inputs

### 3. Observation Pipeline V0

Goal:

- Capture emulator frames in Python
- Crop/downsample/stack them into the policy input format

Exit criteria:

- Python can receive and save short rollouts as frame sequences or tensors

### 4. Episode and Reset Loop

Goal:

- Define exact episode start and end behavior
- Automate repeated episode reset

Exit criteria:

- The environment can run many episodes unattended

### 5. Reward and Success Signals

Goal:

- Implement a reliable notion of progress, failure, and finish

Exit criteria:

- Scripted runs produce clearly better rewards than random runs

### 6. Baselines

Goal:

- Compare random behavior against one or more simple baselines, starting with the easiest reliable option

Exit criteria:

- There is a measurable gap between random and baseline policies

### 7. First RL Training

Goal:

- Train a first agent on the fixed mission setup

Exit criteria:

- The trained policy performs better than random on repeated evaluations

Recommended first algorithm once the environment is stable:

- PPO rather than DQN
- DQN is not the default choice here because the task uses image observations and partial observability

### 8. Scale and Improve

Possible later work:

- better reward shaping
- richer observations
- memory-derived progress signals
- multiple emulator instances
- checkpoint management and logging
- broader Sunshine tasks

## Risks and Unknowns

- Analog steering and controller injection may be harder to automate cleanly than expected.
- Raw-pixel training may require more compute and training time than a structured-state approach.
- Reset speed may be too slow without a good savestate workflow.
- Dolphin scripting/integration may be the hardest part of the project.
- Progress measurement may be difficult without some game-state access.
- Window-focus-dependent input methods may be too flaky for unattended runs.
- Frame capture throughput may become a bottleneck even before training starts.

## Guiding Principles

- Solve emulator control and reset before model training.
- Keep the first task extremely narrow.
- Keep the first action space small.
- Treat reward and reset correctness as first-class engineering problems.
- Only scale complexity after the fixed mission loop is reliable.
- Keep the environment framework-neutral enough to support Gymnasium-compatible tooling and later PPO training.

## Immediate Next Steps

1. Choose the exact emulator integration path.
2. Define the first action space precisely.
3. Decide how progress, failure, and success will be detected in v1, with memory reads allowed if they are the fastest reliable option.
4. Verify that Blooper Surfing Safari can be reset quickly from a savestate.
5. Build the environment loop and baseline evaluation before attempting RL training.


## Similar Project
- The github repo below is similar in problem/solution, however it is coded for mario kart wii specifically so there might not be much carryover in the implementation details.
https://github.com/VIPTankz/Wii-RL 
