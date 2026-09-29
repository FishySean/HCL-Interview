# Design

This is the reasoning that came before the code. Where the implementation
taught me something the sketch got wrong, I have said so rather than editing
history.

## 1. What the problem actually is

The literal requirement is small: a person appears, so wake up, greet, and
wave; the person leaves, so return to idle. A naive version is about fifty
lines.

So the exercise is not the feature. It is whether the structure survives the
four pressures the brief names explicitly — more behaviors, more motions,
motions that fail or overrun, and a traceable control pipeline. I designed
against four questions:

1. Can a new behavior be added without editing anything that already exists?
2. Can a failed or hung motion corrupt the robot's state?
3. Can a bug be reproduced after the fact?
4. Does the system stay correct when the smart part is slow or wrong?

### The non-obvious correctness trap

A detector flickers. A person half-occluded for 200 ms produces
`absent → present`, and a naive system greets them a second time. Presence is
not a per-frame boolean; it is a debounced state over time. I consider this
more important to the user experience than which neural network is used, and
it is the component with the most tests.

## 2. Structure and separation of responsibilities

Five layers, not three. The brief names perception, behavior, and motion; two
more sit *between* them, at the boundaries where the interesting failures live.

```
contracts/        shared vocabulary: events, primitives, verdicts, protocols. Depends on nothing.
runtime/          event bus, injectable clock, tracer, composition root.
perception/       detector -> raw observations. Draws no conclusions.
presence_filter/  raw observations -> debounced, actionable events.
behavior/         FSM + behavior registry + reflexes + policy -> plans.
safety/           robot model + joint limits + self-collision + limb arbitration.
motion/           executor (deadlines, degradation) + backends.
```

One rule makes the decoupling verifiable rather than aspirational: **a layer
may import `contracts` and `runtime`, and never another layer.** Everything is
constructed in one composition root (`runtime/app.py`) and injected downwards.
`tests/test_architecture.py` parses every module's import statements and fails
if this is violated, so the diagram cannot silently stop being true.

| Layer | Owns | Must not own |
| --- | --- | --- |
| Perception | what the sensor saw this frame | whether it means anything |
| Presence filter | whether a person is *really* there, and since when | what to do about it |
| Behavior | what should happen next, given events and state | joint angles, whether a pose is safe |
| Safety | whether a motion is physically permissible, and who owns each limb | why the motion was requested |
| Motion | executing an approved primitive, and the truth about the outcome | whether it should have been approved |

The deeper reason to split perception, behavior, and motion is that they have
different natural frequencies: perception ~30 Hz, behavior event-driven at
~1–5 Hz, motion 50–100 Hz. Fusing them into one loop drags all three to the
slowest.

### Why the presence filter is its own layer

It belongs to neither neighbour. It is not perception: it holds no model and
looks at no pixels. It is not behavior: it decides nothing about what the robot
does. It is the boundary where *noisy* becomes *actionable*.

Giving it a module means the rule "the behavior layer never sees a raw frame"
is enforced by the wiring rather than by discipline — perception publishes
`PersonObserved`, only the filter subscribes to it, and only the filter
publishes `PersonAppeared`. The cost is one extra hop and one event per frame
on the bus, which is the price of a boundary that can be checked by a test.

### Why the safety gate is its own layer

In the first version, safety was implicit: limb arbitration lived inside the
motion executor and joint limits did not exist. I listed "safety as a separate,
non-bypassable layer" as a production improvement. It is now built, because
implicit safety is the kind of thing that stays implicit until an arm swings
into a person.

Non-bypassable is a structural claim, not a convention: behavior publishes
`MotionCommand`, the motion layer subscribes to `ApprovedMotion` and to nothing
else, and only the gate converts one into the other. There is no code path from
a decision to an actuator that skips the checks — not a forgotten call site,
not a behavior written by someone who never read the gate, not a language model
picking an action. A test asserts the subscription topology directly.

The gate answers two different kinds of question, and separating them is the
one subtle decision here:

- **Static** questions are properties of the command alone — does this
  primitive exist, do its goal angles fall inside the joint limits, does the
  goal pose put the robot inside itself. They can be answered the instant the
  command appears, so they run as a pipeline stage and a bad command never
  reaches the motion layer.
- **Dynamic** questions are properties of the moment — is this limb already
  moving, and does the newcomer outrank its owner. The answer can change
  between asking and acting, so acquisition has to be atomic with starting the
  motion. `acquire` is therefore called by the executor, but the ownership
  table and the arbitration policy still live in the gate, which remains the
  single source of truth.

Two rules in the arbitration deserve defending. Acquisition is
**all-or-nothing**: a motion needing both arms takes both or neither, so a
failed attempt never leaves one arm locked. And **equal priority is rejected,
not queued** — two behaviors of the same rank fighting over a limb is
contention to be resolved upstream, not a race won by whoever published last.

### The robot model

Joint names, joint ranges, and limb membership are properties of the machine,
not of the program, so they live in `config/robot.yaml` rather than in code.
The moment a second robot exists they would have to anyway. The file is a
stand-in for a URDF; `RobotModel.from_urdf` is the seam where a real one gets
parsed, and it raises `NotImplementedError` rather than faking it, so nobody
mistakes the stand-in for URDF support.

The model is validated at construction, and every registered primitive is
screened against it at boot. A typo in a joint name stops the robot from
starting instead of surfacing as a denial halfway through a greeting.

## 3. Communication: a closed loop, not a pipeline

My first sketch was a one-way pipeline. Implementing the failure requirements
showed that is not sufficient. "Motion execution may occasionally fail or take
longer than expected" means motion must talk **back**: if a wave fails and the
behavior layer never hears about it, the robot sits in `GREETING` forever.

```
perception ─PersonObserved─▶ presence_filter ─PersonAppeared─▶ behavior
                                                                  │
                                                          MotionCommand
                                                                  │
                                                                  ▼
                                                              safety gate
                                                                  │
                                                          ApprovedMotion
                                                                  │
   behavior ◀── MotionResult{COMPLETED | FAILED | TIMED_OUT ──── motion
                             | PREEMPTED | REJECTED}
```

The transport is a typed async publish/subscribe bus. Every subscriber owns a
queue and runs as its own task, so a slow decision cannot stall the camera
loop. `publish` is synchronous and never blocks; when a queue overflows the
*oldest* event is dropped, because for a robot a stale perception frame is
worth less than a fresh one.

This buys three properties: layers are testable in isolation by publishing a
scripted event sequence; a `trace_id` born with a perception event rides all
the way to the motion result, so one id reconstructs a full causal chain; and
because the event stream is the entire input surface, a session can in
principle be recorded and replayed offline.

## 4. Perception: the ML decision

Person presence is a **detection** problem, not classification.

| Option | Verdict |
| --- | --- |
| Background subtraction (MOG2) | Rejected. A motionless person disappears; breaks on lighting change. |
| Haar cascade / HOG+SVM | Rejected. Poor recall on partial and side views, high false positive rate. |
| Face detector (BlazeFace / SCRFD) | Rejected as *primary* — a person facing away is still present. Useful later as an *engagement* signal. |
| **Lightweight single-stage detector (YOLO11n / MobileNet-SSD)** | **Chosen.** |
| A VLM in the presence path | Rejected. Hundreds of milliseconds, non-deterministic, costly. It belongs in the slow tier. |

Why the lightweight detector:

- Real time (>30 FPS) on modest CPU/edge hardware, INT8-quantizable, deployable
  through ONNX Runtime, TFLite, or CoreML.
- "Person" is COCO class 0, the best-represented class in public detection
  data, so off-the-shelf accuracy is high with **zero data collection**.
- It returns a bounding box, not a flag. The box gives a distance proxy (area)
  and, with a cheap tracker, identity continuity — so the robot greets a
  *person* rather than a *frame*, which is what makes "welcome back" possible.

**Two-tier perception.** The fast tier (detector + tracker, deterministic)
drives the state machine. A slow tier (a VLM, ~0.5 Hz or event-triggered)
enriches context: is the person waving, carrying something, looking at me? The
hard rule is that **the system must be fully correct if the slow tier never
answers**. `SceneContext` is an input to decisions and never a dependency of
them.

## 5. Behavior: smart *and* extensible

Three pieces, in ascending order of smartness and descending order of
trustworthiness.

1. **The state machine** (`behavior/state.py`). The transition table is data,
   not control flow, so it can be reviewed by a person and asserted on
   exhaustively — including a test that no state is a dead end.

2. **The behavior registry** (`behavior/registry.py`). Each behavior is a
   plugin declaring its trigger types, valid states, priority, cooldown, and a
   plan. Gating lives in the registry so every behavior gets it right by
   construction. Adding one requires no edit to the service, the FSM, the
   policy, or the motion layer.

3. **The policy** (`behavior/policies.py`), where a VLM plugs in. Two
   constraints make a language model safe in a decision path:
   - **Constrained output.** A policy may only return a name from the candidate
     list the registry produced. A hallucinated action, a malformed reply, or an
     exception is discarded and the deterministic `RulePolicy` decides instead.
     The model can be wrong; it cannot be dangerous.
   - **A deadline.** A decision that arrives late is wrong even if it is
     correct. `DeadlineArbiter` races the policy against the clock.

   Both constraints live in the arbiter rather than in any policy, which is why
   `behavior/vlm/RealVLMPolicy` — a hosted vision-language model reached over
   HTTPS, optionally with a camera frame attached — could be added without
   changing the protocol, the deadline, the validation, or the fallback. It is
   a fourth implementation of a two-method interface. The worst a bad model can
   do is make the robot boring.

**On the hierarchical output idea.** I agree with the two-level split, and I
put the boundary here: the policy chooses a **primitive name plus parameters**
(`wave(hand=right, amplitude=0.8, speed=1.0)`) and never a joint trajectory.

This is not a stylistic preference, and the safety gate is what makes the
argument concrete. A parameterised primitive resolves to a specific set of goal
joint angles *before anything moves*, so it can be range-checked against the
robot model and tested for self-collision. A generated trajectory cannot be
checked that way — there is nothing to compare it against and no way to unit
test it. A hallucinated primitive name fails a registry lookup; a hallucinated
trajectory reaches a servo.

The coordination pattern between joints — how a shoulder, an elbow, and a wrist
move together to read as a wave — is exactly the kind of structure that should
be learned offline from demonstration data rather than hand-authored. But the
output of that learning belongs in the *parameterisation* of a primitive, fixed
at build time and reviewed like any other code, not generated per invocation at
run time. Learn the shape offline; freeze it as a primitive; expose only its
parameters to the policy.

**Reflexes** (`behavior/reflexes.py`). Waking up when someone walks in is always
safe and always correct, so there is nothing to decide, and deciding would cost
the one thing a greeting cannot spend: the first 200 ms. The reflex is
dispatched immediately while the policy keeps thinking about *how* to greet.
This pays for the latency-versus-expressiveness trade-off structurally rather
than by shortening a timeout, and it sacrifices less than a plain timeout
would. `test_the_reflex_moves_before_the_policy_has_decided` asserts it.

## 6. Motion: executing, and telling the truth

What this layer no longer does is as informative as what it does. Safety
checking and arbitration moved to the gate; the executor is left with
execution.

- **Primitive catalogue** (in `contracts`, because behavior names primitives,
  safety validates them, and motion executes them — it is shared vocabulary).
  Adding a motion is one `register(...)` call.
- **Deadlines**: every primitive runs under a watchdog, so a hang becomes
  `TIMED_OUT` instead of a robot frozen mid-gesture.
- **Degradation ladder**: `wave → nod`. Failing to move should not mean failing
  to interact. Preemption and rejection are *not* retried — those are the
  arbiter working as intended. Crucially, a **substituted primitive is screened
  by the gate exactly like the original**: a fallback is not exempt just
  because the motion it replaces was approved, and parameters carrying over to
  a primitive that moves different joints is a real way to go out of range.
- Failure is always a returned value, never an escaping exception.

## 7. Trade-offs and problems I see

1. **Latency versus accuracy.** Addressed by two-tier perception, a hard
   decision deadline, and speculative dispatch of the safe action. Not
   eliminated: a genuinely better decision that needs 2 s is still unavailable.
2. **Decoupling adds hops and complexity.** A real cost, and the two extra
   layers made it larger: every frame now crosses the bus, and every motion
   crosses it twice. The bus is for cognition; a reflex path must stay short,
   and an emergency stop should not travel through a queue at all — which is
   why the gate is a *permission* layer, not a stop mechanism. See the
   production list.
3. **Event flapping.** Asymmetric hysteresis and cooldowns fix repeated
   greetings at the price of ~1 s of exit latency.
4. **FSM rigidity versus LLM flexibility.** I chose predictability. A robot
   that surprises you is a safety problem, not a personality.
5. **The simulation-to-reality gap.** Simulation cannot model dynamics,
   contact, or collision. What these tests really verify is the abstraction
   boundary, and I would rather say that than overclaim. The self-collision
   check is the sharpest example: it is a hand-written table of joint-space
   boxes that reasons about the goal pose only, knows nothing about the path
   taken to reach it, and was not derived from geometry. It is isolated behind
   one function so that what is being faked is obvious to a reviewer rather
   than buried inside the gate.
6. **Python and asyncio are not hard real time.** Fine for a supervisory layer.
   On a real robot the servo loop belongs in a C++ real-time thread and this
   codebase stays the deliberative layer above it.
7. **VLM cost and privacy.** Camera frames leaving the device is a genuine
   concern. The `Policy` protocol lets an on-device model be substituted
   without touching a single behavior.
