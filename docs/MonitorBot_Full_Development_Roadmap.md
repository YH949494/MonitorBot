# MonitorBot - Full Development Roadmap

# Objective

Build a production-grade desktop slot intelligence system that can:

* auto-detect game regions
* identify multiple slot games
* detect reel grids
* detect symbols/wild/scatter/bonus
* track spin lifecycle
* click spin automatically
* record spin events
* estimate observed behavior patterns
* generate per-game intelligence reports
* scale across providers

Important:
This is NOT only a "slot detector".

Correct architecture target:

```text
Game Vision Intelligence Framework
```

---

# Current Status

## Completed / In Progress

### Phase 1 - Calibration Foundation

Status: Mostly completed

Implemented:

* screen/image capture
* auto calibration
* reels detection
* spin button detection
* popup detection
* grid estimation
* JSON profile storage
* preview overlays
* confidence scoring
* profile validation
* CLI preview

Current limitation:

* no interactive correction UI yet
* no symbol understanding yet
* no spin state understanding yet

Key principle:
Calibration reliability is more important than AI at this stage.

---

# Phase 2 - Interactive Calibration Review UI

## Goal

Allow users to:

* review auto-detected regions
* manually fix bad detections
* confirm/save profiles safely

## Required Features

### UI

* PySide6 desktop UI
* draggable region editing
* overlay rendering
* confidence display
* grid line rendering

### Editable regions

* reels
* spin_button
* popup_close

### Controls

* Re-run Auto Detect
* Confirm Calibration
* Save Profile
* Exit

### Validation

* calibrated=true only after confirmation
* required regions:

  * reels
  * spin_button

## Deliverables

* review.py
* overlay.py
* scaling helpers
* manual override logic
* tests

## Important Notes

Do NOT package EXE before this phase completes.

Without manual correction:
bad calibration = fake analytics.

---

# Phase 3 - Spin State Engine

## Goal

Teach the bot to understand game lifecycle states.

This is the MOST IMPORTANT PHASE after calibration.

## Required States

```text
idle
spin_start
spinning
reel_slowdown
settling
win_animation
bonus_trigger
bonus_mode
popup_open
transition
```

## Required Logic

### Motion analysis

* frame differencing
* motion thresholds
* reel movement intensity

### Stability detection

* stable frames count
* frame similarity scoring

### Transition guards

Prevent:

* duplicate spin reads
* motion blur reads
* bonus misfires

## Required Outputs

Per frame:

```json
{
  "state": "spinning",
  "confidence": 0.92
}
```

## Important Notes

Without state synchronization:

* symbol detection becomes garbage
* RTP estimates become fake
* duplicate counting explodes

Most projects fail here.

---

# Phase 4 - Safe Click Automation

## Goal

Allow controlled auto-spin interaction.

## Features

### Click engine

* click spin button
* randomized timing
* configurable delays

### Safety guards

* only click in idle state
* block double-click spam
* cooldown protection
* emergency stop hotkey

### Humanization

* random click offsets
* random delays
* optional mouse movement curves

## Important Notes

Never click while:

* reels moving
* popup active
* bonus active
* confidence unstable

---

# Phase 5 - Reel Grid Extraction

## Goal

Extract reel symbols cleanly after spin settles.

## Features

### Grid slicing

* convert reels box into cells
* support:

  * 5x3
  * 5x4
  * 6x4
  * 6x5
  * cluster layouts

### Cell extraction

Output:

```json
{
  "reel": 1,
  "row": 2,
  "image": "cell crop"
}
```

### Frame timing

* only extract during stable settled state

## Important Notes

Do NOT classify symbols yet.
Only extract stable cell images first.

---

# Phase 6 - Symbol Classification Engine

## Goal

Identify:

* wild
* scatter
* bonus
* regular symbols

## Stage 1 (Recommended)

Use:

* template matching
* ORB/SIFT
* perceptual hashing

NOT YOLO yet.

## Stage 2

Later:

* CNN
* YOLO
* transformer models

## Required Outputs

```json
{
  "symbol": "scatter",
  "confidence": 0.88
}
```

## Important Notes

Most people overengineer too early.

Deterministic matching is enough initially.

---

# Phase 7 - Multi-Game Profile System

## Goal

Support multiple slot games safely.

## Structure

```text
config/slot_profiles/
```

## Per-profile fields

```json
{
  "game_id": "",
  "provider": "",
  "reel_count": 5,
  "row_count": 3,
  "wild_symbols": [],
  "scatter_symbols": [],
  "bonus_symbols": [],
  "symbol_aliases": {}
}
```

## Required Features

* game identification
* profile switching
* per-game symbol mapping
* per-provider layouts

## Important Notes

Never mix metrics across games.

All analytics must be isolated by:

* game_id
* session_id

---

# Phase 8 - Spin Event Storage

## Goal

Persist all observed spin events.

## Required Model

```json
{
  "game_id": "",
  "session_id": "",
  "spin_id": "",
  "timestamp": "",
  "bet_amount": 0,
  "payout_amount": 0,
  "grid": [],
  "detected_symbols": [],
  "wild_count": 0,
  "scatter_count": 0,
  "bonus_count": 0,
  "bonus_triggered": false,
  "confidence": 0.0
}
```

## Storage Options

Initial:

* SQLite

Later:

* PostgreSQL
* ClickHouse

## Important Notes

Store:

* raw frame references
* symbol confidence
* timestamps

Never store only final metrics.

---

# Phase 9 - Metrics Engine

## Goal

Generate observed gameplay intelligence.

## Required Metrics

### Frequency

* hit frequency
* bonus frequency
* dead spin rate

### Symbol metrics

* symbol weight by reel
* scatter frequency
* wild frequency

### Win metrics

* visual win rate
* real win rate
* max observed multiplier
* longest dry streak

### Behavioral metrics

* inferred volatility
* average spins between bonus

## Important Notes

Never claim:

* true RTP
* true reel strip
* true max exposure

Use:

```text
estimated
observed
inferred
```

ONLY.

---

# Phase 10 - Reporting Layer

## Goal

Generate readable reports.

## Outputs

### CLI

```text
python -m monitorbot.report
```

### Desktop dashboard

* live metrics
* session summary
* volatility charts
* bonus timelines

### Export

* JSON
* CSV
* HTML reports

---

# Phase 11 - Dataset Collection Layer

## Goal

Prepare for ML upgrades later.

## Store

* symbol crops
* labeled symbols
* bonus frames
* spin sequences

## Important Notes

No large-scale ML before dataset exists.

---

# Phase 12 - ML Upgrade Layer

## Goal

Improve:

* symbol detection
* game classification
* sequence prediction

## Future Models

### Detection

* YOLO
* EfficientNet
* ViT

### Sequence analysis

* LSTM
* Temporal transformers

## Important Notes

ML comes LAST.

Most projects fail because they start here first.

---

# Phase 13 - EXE Packaging

## Goal

Package stable system into:

```text
MonitorBot.exe
```

## Packaging

* PyInstaller
* embedded configs
* logging
* updater support

## Required Features

* auto profile loading
* GUI launcher
* portable mode
* crash-safe logging

## Important Notes

EXE packaging is NOT a feature phase.

It is only:

```text
distribution phase
```

Do not package unstable architecture.

---

# Final Architecture

```text
capture
-> calibration
-> review UI
-> spin state engine
-> frame stabilization
-> grid extraction
-> symbol classification
-> spin ingestion
-> storage
-> metrics engine
-> reporting
-> ML upgrades
-> EXE packaging
```

---

# Biggest Technical Risks

## Risk 1 - State desynchronization

Most dangerous issue.

Causes:

* duplicate spins
* fake RTP
* incorrect symbol reads

## Risk 2 - Motion blur classification

Never classify during reel motion.

## Risk 3 - Mixing game profiles

Must isolate:

* game_id
* provider
* session_id

## Risk 4 - Premature ML

Do NOT start with YOLO.

First:

* deterministic pipeline
* stable state engine
* reliable datasets

---

# Production Rules

## Never fake confidence

If uncertain:

* lower confidence
* require manual review

## Never silently fail

Always:

* log reason
* log state
* log confidence

## Never overwrite raw observations

Store:

* original detections
* timestamps
* raw confidence

## Never claim theoretical RTP

Only report:

* observed behavior
* inferred volatility
* estimated symbol frequency

---

# Priority Order

Correct order:

```text
1. Calibration
2. Review UI
3. Spin state engine
4. Click automation
5. Grid extraction
6. Symbol classification
7. Storage
8. Metrics
9. Reports
10. Dataset collection
11. ML upgrades
12. EXE packaging
```

Wrong order:

```text
YOLO
-> AI
-> prediction
-> RTP guessing
```

before stable infrastructure.
