# 5-Minute Project Update

## Slide 1 — Project Title

**Vocal Onset Detection: How well do classical onset detectors work on singing voice?**

Music Information Retrieval Final Project

**Research focus:** onset detection for solo and polyphonic vocals.

### Speaker notes

Hi everyone, today I’m going to give a short update on my final project. My project is about onset detection, but specifically for singing voice. Most onset detection methods are usually discussed in the context of instrumental music or clear percussive attacks, so I want to ask how well these methods work when the sound source is voice, where onsets can be soft, legato, or blurred across multiple singers.

---

## Slide 2 — Research Question

**Main question**

How well do onset detection methods perform on vocals, and how does performance change from solo singing to polyphonic choir singing?

**More specific version**

Across:

- **Solo vocals:** vocadito
- **Polyphonic vocals:** Dagstuhl ChoirSet

I will compare:

- librosa Spectral Flux
- librosa SuperFlux
- madmom CNN onset detector

### Speaker notes

The project question is intentionally narrow and answerable. I’m not trying to build a new onset detector from scratch. Instead, I want to evaluate a few well-known baseline methods and understand their failure modes on vocals. The comparison is between solo singing and choir singing, because I expect choir mixtures to be much harder: several voices may enter at nearly the same time, and the attacks are often smooth rather than percussive.

---

## Slide 3 — Datasets

**Dataset 1: vocadito**

- Solo vocal recordings
- 40 short clips
- Manual note annotations
- Ground truth onsets come from note start times

**Dataset 2: Dagstuhl ChoirSet**

- SATB choir recordings
- Polyphonic room-mic mixes
- 20 annotated multitracks used in this project
- Simultaneous note onsets are merged within 30 ms

### Speaker notes

For the solo condition I’m using vocadito, which contains 40 solo vocal clips with manual note annotations. For the polyphonic condition I’m using Dagstuhl ChoirSet, which contains choir recordings. I use the room-mic mix as the audio input, because that represents the real polyphonic vocal scenario. The annotations are note onsets from the multitrack score, and if multiple singers start almost simultaneously, I merge those onsets within 30 milliseconds so they count as one perceived onset event.

---

## Slide 4 — Methods and Evaluation

**Three baseline detectors**

1. **Spectral Flux** — detects sudden spectral change
2. **SuperFlux** — spectral flux with vibrato suppression
3. **madmom CNN** — pre-trained neural onset detector

**Evaluation**

- Metric: Precision, Recall, F-measure
- Matching tolerance: ±50 ms
- Tool: `mir_eval.onset.f_measure`

**Qualitative analysis**

- Spectrogram overlays
- False positives vs. missed onsets
- Error categories: vibrato false alarms, legato misses, double triggers

### Speaker notes

The three methods cover a useful range. Spectral Flux is the classic signal-processing baseline: if the spectrum changes suddenly, it predicts an onset. SuperFlux is a related method but includes vibrato suppression, which is important for voice because vibrato can create fake spectral changes. The third method is the madmom CNN onset detector, which is a pre-trained neural model.

For evaluation, I use the standard MIR onset detection metric: a prediction is correct if it falls within 50 milliseconds of a reference onset. I will report precision, recall, and F-measure. I also plan to do qualitative error analysis by plotting spectrograms with reference and predicted onset markers.

---

## Slide 5 — Expected Challenges and Deliverables

**Expected challenges**

- Vocal onsets can be soft and gradual
- Legato transitions may have no clear energy burst
- Vibrato can create false positives
- Choir mixtures blur individual singer onsets

**Deliverables**

- Reproducible GitHub repository
- Dataset loaders
- Three baseline detectors
- Quantitative evaluation table
- Notebook demo with visual error analysis

### Speaker notes

My expectation is that vocals will expose weaknesses in standard onset detection. In solo singing, I expect neural methods to do reasonably well, but still struggle with soft legato transitions. In choir singing, I expect performance to drop because the attacks are distributed across singers and the mixture is less transient-like.

The final deliverable will be a reproducible GitHub repository with data loaders, detector implementations, evaluation code, and a self-contained notebook demo. The notebook will include both the quantitative results and visual examples of where each method succeeds or fails.

---

# Full 5-Minute Script

Hi everyone, today I’m going to give a short update on my final project. My project is about onset detection, but specifically for singing voice.

Onset detection means finding the starting time of each note or sound event. This is a common task in Music Information Retrieval, but many standard onset detectors are designed around music with clear attacks, like drums, piano, or other instrumental sounds. Singing voice is different. Vocal notes can begin gradually, singers often connect notes with legato, and vibrato can create spectral changes that look like onsets even when there is no new note.

So my main research question is: how well do existing onset detection methods work on vocals? More specifically, I want to compare performance between solo singing and polyphonic choir singing.

For datasets, I plan to use two sources. The first is vocadito, which contains 40 short solo vocal recordings with manual note annotations. The note start times will serve as ground-truth onsets. This gives me a clean solo-vocal condition.

The second dataset is Dagstuhl ChoirSet. This dataset contains multitrack SATB choir recordings. For this project I will use the annotated choir multitracks and evaluate on the room-mic mix, which is a realistic polyphonic vocal signal. Since several singers can start at almost the same time, I will merge simultaneous or near-simultaneous score onsets within 30 milliseconds, so that they count as one perceived onset event.

For methods, I will compare three baselines. First, Spectral Flux from librosa, which detects sudden frame-to-frame changes in the spectrum. Second, SuperFlux, also from librosa, which is similar but uses a maximum filter to suppress vibrato-related false positives. Third, I will use madmom’s pre-trained CNN onset detector, which represents a neural-network-based baseline.

The quantitative evaluation will use the standard onset detection metric from `mir_eval`: precision, recall, and F-measure with a 50 millisecond tolerance window. So if a predicted onset is within 50 milliseconds of a reference onset, it counts as correct.

In addition to the numbers, I also want to do qualitative error analysis. I will plot spectrograms and overlay the ground-truth onsets and predicted onsets. This should help identify common failure cases, such as false positives caused by vibrato, missed onsets during legato transitions, and double triggers around consonants.

My expectation is that solo vocals will be easier than choir vocals. In solo singing, the CNN detector may perform well because the signal is relatively clean. But in choir mixtures, individual onsets may be blurred together, so I expect a significant drop in performance. I am especially interested in whether SuperFlux is more robust than plain Spectral Flux for vocal vibrato, and whether the CNN becomes too conservative on polyphonic choir recordings.

The final deliverable will be a reproducible GitHub repository. It will include dataset loading code, detector implementations, evaluation scripts, result tables, and a self-contained Jupyter notebook demo. The notebook will show the main quantitative comparison and several visual examples for error analysis.

That’s the current plan for my project. The main goal is not only to report which detector gets the best F-measure, but also to explain why vocal onset detection is hard and what kinds of vocal sounds cause different methods to fail.
