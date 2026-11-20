# OpenMind LSL stream specification

**Version 1.1.0**

This document is the contract between producers of EEG streams (OpenMind Plug,
or any third-party tool) and their consumers (OpenMind Core, Sandbox, recorders,
visualizers). A stream that follows it can be used by every OpenMind component
without configuration.

The key words **MUST**, **SHOULD** and **MAY** are used as in RFC 2119.
Field names follow the [XDF meta-data conventions](https://github.com/sccn/xdf/wiki/Meta-Data)
where they exist, so standard tools (LabRecorder, pyxdf, MNE) read them natively.

A producer publishes **two LSL streams**: the EEG stream and the markers stream.
`openmind-plug check-stream` checks a live stream against this document.

## 1. EEG stream

### 1.1 Stream header

| Field | Value |
|---|---|
| `name` | **MUST** be `openmind-eeg` |
| `type` | **MUST** be `EEG` |
| `channel_count` | number of channels, **MUST** be ≥ 1 and equal to the number of `channel` entries in `desc` |
| `nominal_srate` | sampling rate in Hz, **MUST** be > 0 (regular stream) |
| `channel_format` | **MUST** be `float32` |
| `source_id` | **SHOULD** be stable across restarts of the same device, so consumers reconnect automatically. Plug uses `openmind-plug:<driver>:<model>` |

### 1.2 Samples

- Values are in **microvolts** (µV), for every channel.
- One LSL sample holds one value per channel, in the order of the `channel` entries.
- Missing values (e.g. a lost packet) **MUST NOT** be invented: the producer skips
  them, and consumers detect the gap from the timestamps.

### 1.3 Timestamps

- Timestamps **MUST** come from the LSL clock (`lsl_local_clock()` /
  `pylsl.local_clock()`) of the producing machine, never from wall-clock time.
- Each sample carries the time at which it was **acquired** (best estimate),
  not the time it was sent. Producers that know the exact sampling period
  **SHOULD** send regular timestamps (`t0 + i / srate`).
- Consumers **SHOULD** enable LSL clock synchronization (`proc_clocksync`) to map
  timestamps to their own clock.

### 1.4 Metadata (`desc`)

```xml
<desc>
  <spec_version>1.1.0</spec_version>
  <channels>
    <channel>
      <label>C3</label>
      <unit>microvolts</unit>
      <type>EEG</type>
      <location>          <!-- optional -->
        <X>-65.4</X>
        <Y>-11.6</Y>
        <Z>64.4</Z>
      </location>
    </channel>
    <!-- one <channel> per channel, in sample order -->
  </channels>
  <acquisition>
    <manufacturer>OpenMind</manufacturer>
    <model>synthetic</model>
    <driver>synthetic</driver>
    <software>openmind-plug</software>
    <software_version>0.1.0</software_version>
  </acquisition>
  <reference>Cz</reference>
  <montage>A</montage>
</desc>
```

| Element | Rule |
|---|---|
| `spec_version` | **MUST** be present. Consumers **MUST** accept any `1.x.y` and **MUST** reject another major version. |
| `channels/channel/label` | **MUST** be present, non-empty and unique (case-insensitively). 10-10 labels (`C3`, `Cz`, `Fp1`) **SHOULD** be used with their canonical spelling. |
| `channels/channel/unit` | **MUST** be `microvolts`. |
| `channels/channel/type` | **MUST** be one of `EEG`, `EOG`, `EMG`, `ECG`, `MISC`. |
| `channels/channel/location/{X,Y,Z}` | **MAY** be present: electrode position in **millimeters**, MNE head frame (X towards the right ear, Y towards the nose, Z towards the vertex). Plug fills it for 10-10 labels. |
| `acquisition/manufacturer`, `model`, `driver` | **MUST** be present (`unknown` if not known). `driver` names the producer's driver. |
| `acquisition/software`, `software_version` | **SHOULD** name the producing software and its version. |
| `reference` | **MUST** be present: reference electrode or scheme (`Cz`, `average`, `left mastoid`, `none`, `unknown`). |
| `montage` | **MUST** be present: montage name (`A`, `B`, `C`), or empty if the channels do not follow a named montage. |

Consumers **MUST** ignore unknown elements, so minor versions can add fields.

## 2. Markers stream

### 2.1 Stream header

| Field | Value |
|---|---|
| `name` | **MUST** be `openmind-markers` |
| `type` | **MUST** be `Markers` |
| `channel_count` | **MUST** be `1` |
| `nominal_srate` | **MUST** be `0` (irregular) |
| `channel_format` | **MUST** be `string` |
| `source_id` | **SHOULD** be the EEG `source_id` followed by `:markers` |

`desc` **MUST** contain `<spec_version>` (same rules as above).

### 2.2 Content

Each sample is **one JSON object**, serialized as a compact UTF-8 string. It
**MUST** have a string field `type`; other fields depend on the type.

| `type` | Fields | Example |
|---|---|---|
| `cue` | `label`, `trial` (int, from 1) | `{"type":"cue","label":"left","trial":12,"source":"replay"}` |
| `state` | `label` (`rest`, `left`, `right`, `feet`) | `{"type":"state","label":"left","source":"synthetic"}` |
| `eyes` | `label` (`open`, `closed`) | `{"type":"eyes","label":"closed","source":"synthetic"}` |
| `artifact` | `label` (`blink`, `muscle`, `jaw_clench`...); for `jaw_clench`, `count` (int, number of clenches: 1 or 2) | `{"type":"artifact","label":"jaw_clench","count":2,"source":"synthetic"}` |

- `source` **SHOULD** name the producer of the marker (`synthetic`, `replay`,
  `sandbox`...).
- Consumers **MUST** ignore unknown types and fields.
- The timestamp of a marker **MUST** be on the same LSL clock as the EEG
  stream, equal to the timestamp of the first EEG sample the event applies to.
  Markers produced by another machine (e.g. Sandbox cues) rely on LSL clock
  synchronization.

## 3. Versioning

This specification follows [Semantic Versioning](https://semver.org):

- **patch** (1.0.x): wording and clarifications, no change for producers or consumers;
- **minor** (1.x.0): new optional fields or marker types; existing consumers keep working;
- **major** (x.0.0): any breaking change. Consumers reject unknown major versions.

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | 2026-11-18 | First version. |
| 1.1.0 | 2026-11-20 | Montage `C` (eight motor electrodes); `jaw_clench` artifact marker with `count`. |
