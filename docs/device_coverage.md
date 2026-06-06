# Device Coverage Map

This maps **every device from your original list** onto the assistant's architecture, so nothing is left ambiguous. The principle: anything connectable becomes a **connector** (an MCP server or a custom API client) that the agent calls as a tool. The pattern is identical for every device — what differs is how connectable each one is in reality.

Devices fall into five tiers. Read the tier and you know what's possible.

---

## Tier 1 — Genuinely open (real APIs, full connectors)

These have proper APIs. They become first-class tools the agent can both read from and act on. This is where "automation" is fully real.

| Device | How it connects | Read | Control | Phase |
|---|---|---|---|---|
| Smart home devices | Home Assistant → one tool | Yes | Yes | Devices |
| Cloud storage | Provider API (or MCP server) | Yes | Yes | Services |
| Wi-Fi modems & routers | Vendor API (varies by brand) | Often | Sometimes | Devices |
| Refrigerator *(if smart)* | Vendor cloud (SmartThings / ThinQ / Home Connect) | Yes | Limited | Devices |
| Washing machine *(if smart)* | Vendor cloud | Yes | Limited* | Devices |
| Microwave *(if smart)* | Vendor cloud | Yes | Limited* | Devices |
| Induction cooktop *(if smart)* | Vendor cloud | Yes | Limited* | Devices |
| Electric vehicle (EV) | Brand's vehicle API | Brand-dependent | Brand-dependent | Devices |
| Vehicle (general, if connected) | Brand's API or OBD-II dongle | Some | Some | Devices |

\* Heating appliances usually **block remote start** for safety, by design. Realistic goals: read status, get notifications, switch standby power. Not "run a cook cycle from your phone."

---

## Tier 2 — Sandboxed (hooks & signals, not full control)

Phones, laptops, and tablets are locked down by their operating systems on purpose. You don't "control" them like a smart plug — instead they act as the assistant's **interface and sensors**.

| Device | How it connects | What you get | Phase |
|---|---|---|---|
| Smartphone | iOS Shortcuts / Android Tasker + push notifications | Trigger the agent, receive its output, send location/presence | Interface |
| Laptop / tablet | OS scripts, local agent, notifications | Run local automations, surface the assistant | Interface |

So they *are* covered — just as the surfaces you talk to the assistant through and as sources of context (location, "I'm home"), not as devices it puppeteers.

---

## Tier 3 — Locked ecosystems (input/output surfaces only)

Smartwatches, earbuds, and e-readers talk **only to their parent brand's app**. There's essentially no third-party control surface — no architecture changes this, because the manufacturer never built the door.

| Device | Role in the system | How |
|---|---|---|
| Smartwatch | Output surface | Notifications relayed via the phone |
| Wireless earbuds | Input/output surface | Voice in / audio out via the phone |
| E-reader | (mostly out of reach) | At best, content sync through a cloud API if one exists |

These are covered as *channels*, not as automation targets. That's the ceiling for everyone, not just this build.

---

## Tier 4 — "Dumb" devices (bridged with added hardware)

A **non-smart** microwave, fridge, washer, or cooktop has no network and no computer — there are literally no bits to send. The DIY world bridges these by adding hardware, which then appears in Home Assistant as a normal tool.

| Bridge | What it enables | Caution |
|---|---|---|
| Smart plug | On/off + power-draw monitoring (infer "cycle finished") | Safe, easy starting point |
| ESP32 / microcontroller | Custom sensing/switching wired into a device | Mains voltage = shock/fire risk; voids safety certification |
| IR blaster | Mimic a remote for anything with one | Limited to remote-controllable functions |

Realistic goal for dumb heating appliances: **switch and monitor**, not fully operate — same safety reason as Tier 1.

---

## Tier 5 — Not endpoints (plumbing, consumed as signals)

These aren't things you connect *to* — they're how things connect or where things are. They show up as **inputs and transports**, not as tools.

| Item | What it really is | How the system uses it |
|---|---|---|
| Bluetooth | A transport | The hub uses it to reach BLE sensors/devices; BLE presence as a trigger |
| GPS | A location signal | Consumed (from the phone) as context: location, geofence triggers |

---

## What "covering every device" actually means

Putting it together:

- **Tiers 1 & 4** are full automation targets — each is one more connector, built with the same pattern, sequenced across the Devices phase.
- **Tier 2** devices are your interfaces and sensors — covered, but as surfaces, not puppets.
- **Tier 3** devices are notification/voice channels — covered as far as anyone can cover them.
- **Tier 5** items are signals and transports the other connectors use — covered, just not as endpoints.

So the architecture *does* account for everything. The MVP starts with two or three Tier-1 service connectors only because that proves the pattern cheaply. Every device after that is the same move repeated: build the connector, expose it as a tool, let the agent use it — within the physical and vendor limits that no software design can remove.

---

## Suggested device-connector order (for the Devices phase)

1. Smart home devices via Home Assistant (widest payoff, real control).
2. Cloud storage (easy, useful for memory ingestion — can come earlier).
3. Router (network status, presence).
4. One connected appliance (prove the vendor-cloud pattern).
5. EV / vehicle (if your brand exposes a usable API).
6. Bridged "dumb" devices (smart plugs first, microcontrollers later).
7. Locked-ecosystem channels (watch notifications, earbud voice) — whenever the interface work happens.
