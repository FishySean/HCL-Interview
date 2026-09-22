# Interactive Robot Runtime

A small interactive robot: it notices a person, wakes up, greets them, waves,
and settles back to idle when they leave. Hardware is simulated; the seams
where real hardware attaches are explicit.

The reasoning behind the design is in [DESIGN.md](DESIGN.md). This file
describes the architecture as built, how to run it, and how to extend it.

## Quick start

```bash
pip install -r requirements.txt         # PyYAML (robot model) + pytest
python3 -m robot_runtime.cli            # scripted visit, ~22s
python3 -m pytest tests/ -q             # 77 tests, ~0.3s
```

```
  model: demo_interactive_robot  limbs: head, left_arm, right_arm, torso, voice
  presence filter: enter=3 frames, exit=15 frames
==============================================================================
[  2.15s] SENSE  PersonAppeared  trace=70f0d4e2
[  2.15s] STATE  idle -> waking   (reflex:wake_on_person)
[  2.15s] DECIDE greet_visitor  via trivial in 0ms  considered=['greet_visitor']
[  2.15s] STATE  waking -> greeting   (greet_visitor:step)
  [motion] wake_up() ~0.6s
  [voice] "Hello there! Nice to meet you."
[  4.00s] MOTION say: completed  (say done in 1.85s)
  [motion] wave(hand=right, amplitude=0.8, speed=1.0, cycles=3) ~1.8s
[  5.80s] MOTION wave: completed  (wave done in 1.80s)
[  5.80s] STATE  greeting -> engaged   (greet_visitor:done)
...
[ 13.25s] SENSE  PersonAppeared  trace=69db2b39
[ 13.25s] STATE  idle -> waking   (reflex:wake_on_person)
  [motion] wake_up() ~0.6s
[ 13.37s] DECIDE greet_returning_visitor  via mock-vlm in 121ms  considered=['greet_returning_visitor', 'greet_visitor']
  [voice] "Welcome back! Good to see you again."
...
[ 19.23s] MOTION lower_arms: completed  (lower_arms done in 0.50s)
[ 19.23s] MOTION idle_breathe: preempted  (preempted by a higher-priority command)
[ 20.23s] STATE  disengaging -> idle   (return_to_idle:done)
```

The scripted scenario is chosen to show four things: a brief occlusion at
4.0–4.3 s that the presence filter absorbs, so it must **not** end the visit;
the reflex dispatching `wake_up` at 13.25 s *before* the policy answers at
13.37 s; a real arbitration between two competing greetings once the visitor is
recognised as returning; and a departure preempting whatever gesture is still
running.

### Other ways to run it

```bash
python3 -m robot_runtime.cli --mode manual          # drive it yourself: p / l / q
python3 -m robot_runtime.cli --motion-failure-rate 0.4 --motion-slow-rate 0.3
python3 -m robot_runtime.cli --vlm-invalid-rate 0.5 --vlm-error-rate 0.3
python3 -m robot_runtime.cli --no-vlm               # deterministic rule policy only
python3 -m robot_runtime.cli --trace-file trace.jsonl
python3 -m robot_runtime.cli --robot-config my_robot.yaml
python3 -m robot_runtime.cli --mode camera          # pip install -r requirements-camera.txt
```

Under 40% actuator failure and 30% overrun the robot still ends in `idle`:
motions fail, the fallback ladder runs, the departure preempts a wave in
flight, and the state machine lands cleanly.

## Architecture

Five layers. The brief names three; two more sit *between* them, at the
boundaries where the interesting failures live.

```
  camera          ┌────────────┐  PersonObserved  ┌──────────────────┐
  or script ─────▶│ perception │─────────────────▶│ presence_filter  │
                  └────────────┘   (raw, 20Hz)    │  debounce +      │
                                                  │  hysteresis      │
                                                  └────────┬─────────┘
                                                           │ PersonAppeared
                                                           │ PersonLeft
                                                           ▼
  ┌────────────┐   ApprovedMotion   ┌──────────┐  MotionCommand  ┌──────────┐
  │   motion   │◀───────────────────│  safety  │◀────────────────│ behavior │
  │ deadlines  │                    │  joint   │                 │ FSM +    │
  │ degradation│                    │  limits, │                 │ registry │
  └─────┬──────┘                    │  limbs,  │                 │ + reflex │
        │                           │ collision│                 │ + policy │
        │                           └──────────┘                 └────▲─────┘
        │                                                             │
        └──────────────── MotionResult ───────────────────────────────┘
```

```
robot_runtime/
├── contracts/        events, primitives, safety verdicts, protocols — depends on nothing
├── runtime/          event bus, injectable clock, tracer, composition root (app.py)
├── perception/       detectors (scripted / manual / YOLO) → raw observations
├── presence_filter/  debounce + hysteresis → actionable presence events
├── behavior/         state machine, behavior registry, reflexes, policies
├── safety/           robot model, joint limits, self-collision, limb arbitration
├── motion/           executor (deadlines + degradation), backends
├── config/robot.yaml the robot model (stand-in for a URDF)
└── cli.py            runnable demo
```

**The one structural rule:** a layer imports `contracts` and `runtime`, and
never another layer. `tests/test_architecture.py` parses every module's import
statements and fails if this is violated, so the diagram above cannot silently
stop being true. All wiring happens in `runtime/app.py`, so swapping a scripted
detector for a webcam, simulated actuators for real ones, the collision stub
for a geometric check, or the mock VLM for a real endpoint is a change to that
one file.

### Communication

A typed async publish/subscribe bus. Each layer runs as its own task with its
own queue, because the layers have different natural frequencies (perception
~30 Hz, behavior event-driven, motion 50–100 Hz) and fusing them into one loop
would drag all three to the slowest. `publish` never blocks; a full queue drops
the *oldest* event, since a stale perception frame is worth less than a fresh
one.

The loop is closed, not a one-way pipeline: every `MotionCommand` produces a
`MotionResult` that comes back to the behavior layer. That is what stops a
failed wave from stranding the robot in `GREETING` forever.

### Perception

`PersonDetector` (one call, one frame) → `PersonObserved`. This layer reports
what the sensor saw and draws no conclusions; all temporal reasoning lives
downstream, so the detector can be swapped without any of it moving too.

The ML choice is a lightweight single-stage person detector (YOLO11n class):
real time on modest hardware, "person" is the best-represented class in public
detection data, and a bounding box gives both a distance proxy and — with the
tracker — identity continuity. The full comparison against background
subtraction, Haar/HOG, face detection, and a VLM is in
[DESIGN.md §4](DESIGN.md) and restated at the top of
`perception/detectors/yolo.py`.

### Presence filter

The component that matters most is not the network. `PresenceDebouncer` applies
**asymmetric hysteresis** — enter after 3 confident frames (~150 ms, feels
instant), leave only after 15 missing frames (~750 ms at 20 fps, survives
occlusion) — so a flickering detector cannot produce a second greeting.

It is a layer rather than a helper because it belongs to neither neighbour: it
holds no model and looks at no pixels, and it decides nothing about what the
robot does. Making it a stage means "the behavior layer never sees a raw frame"
is enforced by the wiring — only the filter subscribes to `PersonObserved`, and
only the filter publishes `PersonAppeared`. The debouncer itself is pure and
synchronous, which is why the flapping edge cases are cheap to test
exhaustively.

### Safety gate

Sits between behavior and motion and is the last thing before an actuator. It
loads `config/robot.yaml` — a stand-in for a URDF carrying joint names, joint
ranges, and limb membership — and checks every motion before it runs:

- **Joint limits.** A primitive's parameters resolve to goal joint angles
  (`wave(hand=right, amplitude=0.8)` → `right_shoulder_pitch=1.12`), which are
  range-checked against the model. `amplitude=2.0` asks for 2.80 rad against a
  2.10 rad limit and is refused.
- **Self-collision.** A placeholder, and labelled as one: a hand-written table
  of joint-space boxes that reasons about the goal pose only. See
  `safety/collision.py` for exactly what a real implementation must add.
- **Limb ownership.** One active motion per limb, arbitrated by priority.
  Strictly higher preempts; equal or lower is refused. Acquisition is
  all-or-nothing, so a two-armed motion never leaves one arm locked.

**Non-bypassable** is structural, not a convention: behavior publishes
`MotionCommand`, the motion layer subscribes to `ApprovedMotion` and to nothing
else, and only the gate converts one into the other. A test asserts that
subscription topology directly.

The model is validated at load, and every registered primitive is screened
against it at boot, so a joint typo stops the robot from starting rather than
surfacing as a denial halfway through a greeting.

### Behavior

- **`state.py`** — the transition table, as data. A test asserts no state is a
  dead end.
- **`registry.py`** — behaviors as plugins. Trigger types, valid states,
  priority, and cooldown are declared; the registry does the gating.
- **`reflexes.py`** — the fast path. `wake_up` is dispatched the instant a
  person is confirmed, before any deliberation, because it is always safe and
  always correct. This is how the latency budget is paid structurally instead
  of by shortening a timeout.
- **`policies.py`** — `RulePolicy` (deterministic floor), `MockVLMPolicy`
  (injectable latency, hallucination, and failure), and `DeadlineArbiter`,
  which races the policy against the clock and **validates that the answer is a
  registered behavior name**. A model can be wrong here; it cannot be dangerous.

### Motion

- **`contracts/primitives.py`** — the catalogue. It lives in `contracts`
  because behavior *names* primitives, safety *validates* them, and motion
  *executes* them; it is shared vocabulary. A primitive declares its limbs,
  duration, timeout, fallback ladder, and the goal joint angles its parameters
  map to — not a servo trajectory.
- **`executor.py`** — a watchdog on every primitive and graceful degradation
  (`wave → nod`). A substituted primitive is screened by the gate exactly like
  the original: a fallback is not exempt just because the motion it replaces
  was approved. Preemption and rejection are deliberately *not* retried.
- **`backends/simulated.py`** — injectable failure rate and overrun rate with a
  seeded RNG, so "motion occasionally fails" is a reproducible test fixture.

#### Primitives are parameterised, never generated

A primitive is a name plus parameters: `wave(hand, amplitude, speed)`. The
policy may choose the name and the parameters; it may never emit joint angles
or a trajectory.

The safety gate is what makes this more than a style preference. A
parameterised primitive resolves to specific goal joint angles *before anything
moves*, so it can be range-checked and collision-tested. A generated trajectory
cannot be — there is nothing to check it against. A hallucinated primitive name
fails a registry lookup; a hallucinated trajectory reaches a servo.

The coordination pattern between joints — how shoulder, elbow, and wrist move
together to read as a wave — is exactly the kind of structure to **learn
offline from demonstration data**. But the output of that learning belongs in
the *parameterisation* of a primitive, frozen at build time and reviewed like
any other code, rather than generated per invocation at run time. Learn the
shape offline, freeze it as a primitive, expose only its parameters.

### Tracing

A `trace_id` is created with the perception event that caused a reaction and
carried unchanged through the decision, every command, and every result:

```bash
python3 -m robot_runtime.cli --trace-file trace.jsonl
grep 5342e76c trace.jsonl
```

`runtime/tracing.py` is one method wide so it can be re-pointed at
OpenTelemetry without touching a call site.

## Extending it

**A new motion** — one `register` call. The joint targets are what let the
safety gate range-check it before it ever runs:

```python
primitives.register(Primitive(
    name="bow", limbs=("torso",),
    duration_s=0.8, timeout_s=2.0, fallbacks=("nod",),
    targets=(JointTarget("torso_pitch", 0.30, scales_with="depth"),),
    defaults={"depth": 1.0},
))
```

**A new behavior** — one class:

```python
class BowToVisitor(Behavior):
    name = "bow_to_visitor"
    priority = Priority.URGENT
    triggers = (PersonAppeared,)
    valid_states = (RobotState.IDLE, RobotState.WAKING)
    success_state = RobotState.ENGAGED
    failure_state = RobotState.ENGAGED

    def plan(self, context):
        return [PlanStep(command=self.command("bow", context.event.trace_id),
                         enter_state=RobotState.GREETING)]

behaviors.register(BowToVisitor())
```

Neither requires an edit to the service, the state machine, the policy, or the
motion layer. `test_a_new_behavior_and_a_new_motion_need_no_change_to_the_core`
registers both from outside the package and asserts they run.

## Tests

77 tests, ~0.3 s.

Every test runs on `FakeClock`, a virtual clock driven by explicit `advance`
calls. A 750 ms presence timeout or a 10 000 s motion hang costs no wall-clock
time and cannot flake on a loaded CI machine. A 5-second real-time guard in
`conftest.py` turns any genuine deadlock into a fast failure.

| File | What it pins down |
| --- | --- |
| `test_presence_filter.py` | Flicker, occlusion, low confidence. One appearance produces exactly one event, and a flickering stream produces none at all. |
| `test_state_machine.py` | The transition table is complete, has no dead ends, and refuses illegal moves. |
| `test_safety_gate.py` | Joint limits (including following the `hand` parameter), self-collision, unknown primitives, missing parameters; and on arbitration: a lower- or equal-priority motion cannot take a busy limb, a higher one can and names its victim, multi-limb acquisition is all-or-nothing, and a preempted motion cannot release its successor's limb. |
| `test_motion_executor.py` | Timeout, failure→fallback, fallbacks being re-screened, preemption, rejection, concurrent limbs, limb release after every outcome. |
| `test_policy.py` | Late answers, hallucinated actions, and crashes each degrade to the deterministic policy. |
| `test_architecture.py` | No layer imports another layer; `contracts` imports nothing; only the composition root knows all five; the motion layer cannot subscribe to unapproved commands. |
| `test_end_to_end.py` | Wake/greet/wave then idle; no double greeting after occlusion; reflex precedes the decision; a failing wave degrades to a nod; departure preempts a gesture; a misparameterised behavior is stopped by the gate rather than by the hardware; one trace id spans the whole pipeline; new behaviors and motions plug in. |

## What I would improve for production

The honest list, roughly in the order I would do it.

1. **Move the control loop out of Python.** `asyncio` is right for the
   deliberative layer and wrong for servo control. In production the motion
   primitives become a real-time C++ node (ROS 2 / micro-ROS) and this codebase
   becomes the supervisor above it, talking over DDS instead of an in-process
   bus. The `MotionBackend` protocol is where that substitution happens.

2. **Finish the safety layer.** The gate exists and is non-bypassable, but it
   is a *permission* layer, not a stop mechanism, and the distinction matters.
   Production needs: real self-collision checking (sweep the URDF collision
   meshes through forward kinematics along the whole trajectory, not just the
   endpoint, with FCL or Bullet, including the environment and the person);
   velocity, acceleration, and torque limits rather than position limits alone;
   workspace bounds; and an emergency stop implemented as an independent
   watchdog process that can cut power regardless of what any of this code
   decided — an e-stop must never travel through an event queue.

3. **Parse the real URDF.** `config/robot.yaml` restates joint limits that
   already exist in the robot description, which means they can drift apart.
   `RobotModel.from_urdf` is the seam; it currently raises `NotImplementedError`
   rather than faking support.

4. **Validate the perception tier on real data.** The ML argument here is a
   reasoned selection, not a measured result. Production needs a labelled
   evaluation set from the robot's actual deployment (its lighting, its camera,
   its mounting height), a precision/recall target, and continuous monitoring —
   detection quality degrades quietly as an environment changes.

5. **Tune the debouncer against recorded sessions, not intuition.** The
   enter/exit thresholds are currently defensible guesses. They should be fit to
   recorded footage against a cost function that weighs a missed greeting
   against a duplicated one.

6. **Make the VLM path real, and measure it.** A production policy needs
   schema-constrained decoding, a token and latency budget per decision, a cost
   ceiling, caching for repeated contexts, and a fallback that is exercised in
   production rather than only in tests. Privacy pushes toward an on-device
   model; the `Policy` protocol already allows the swap.

7. **Persist and replay the event stream.** The bus already carries everything
   the system knows. Writing it to disk and adding a replay harness turns "the
   robot did something weird yesterday" into a reproducible test case, and
   builds the dataset needed for item 3.

8. **Real observability.** Swap the tracer for OpenTelemetry, export spans and
   metrics (decision latency percentiles, fallback rate, motion failure rate by
   primitive, safety denials by reason, dropped-event count), and alert on
   them. The gate already counts denials by reason and the bus already counts
   drops; nothing currently looks at either number. A rising JOINT_LIMIT
   denial rate means a behavior is misparameterised and should page someone.

9. **Behavior authoring for non-programmers.** Once there are thirty behaviors,
   Python subclasses stop being the right interface. A declarative manifest with
   a schema, plus a linter that catches unreachable triggers and conflicting
   priorities, scales better than code review.

10. **Multi-person and identity.** The system currently reasons about "a
   person". Real deployments need tracking across multiple people, choosing who
   to engage, and — with explicit consent and a retention policy — remembering
   returning visitors beyond a 60-second window.

11. **Hardware-in-the-loop CI.** The simulation tests the abstraction boundary,
    not the dynamics. A nightly run on real hardware, plus a physics simulator
    (MuJoCo / Isaac) in between, is what would catch the class of bug this test
    suite structurally cannot.
