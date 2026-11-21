# Recording

`openmind-plug record` saves the standard LSL stream (EEG and markers) to a FIF
file that MNE, EEGLAB-compatible tools and `ReplaySource` can read. Markers come
from **every** `openmind-markers` stream, including those of other programs that
start during the recording, such as the calibration scenes of OpenMind Sandbox
(`trial_start/1`, `cue/left`, `trial_end/1`, `event/block_start`...).

> [!IMPORTANT]
> EEG recordings are **health data**. Subjects are identified only by a pseudonym
> (`P01`, `P02`...): the recorder refuses anything else. Files are written outside
> the repositories, in `$OPENMIND_DATA_DIR/recordings/`, and must never be committed.

## From the command line

```bash
# terminal 1: a source
openmind-plug stream --source synthetic

# terminal 2: record it for one minute
openmind-plug record --subject P01 --duration 60
```

| Option | Meaning |
|---|---|
| `--subject` | Pseudonym, `P` followed by at least two digits (required) |
| `--out` | Output file ending with `_raw.fif` (default below) |
| `--duration` | Seconds to record (default: until Ctrl+C) |
| `--source-id` | Pick one producer when several stream |
| `--overwrite` | Replace an existing file |

The default file is
`$OPENMIND_DATA_DIR/recordings/<subject>/<subject>_<YYYYmmdd-HHMMSS>_raw.fif`.
A warning is printed if the output would be inside a Git repository.

## From Python

```python
from openmind_plug.recorder import Recorder, record_source, record_stream
from openmind_plug.sources.synthetic import SyntheticSource

recorder = record_source(SyntheticSource(seed=0), duration=30, subject="P01")
path = recorder.save()  # default location

recorder = record_stream("P01", duration=30)  # from the live LSL stream
```

`Recorder` can also be fed by hand with `add(samples, timestamps)` and
`add_markers(markers)`, then saved with `save(path)` or converted with `to_raw()`.

## File content

| Content | Stored as |
|---|---|
| Samples | Volts (MNE convention), double precision: microvolt values come back unchanged |
| Channels | Labels and types (EEG, EOG, EMG, ECG, MISC), 10-10 positions in the montage |
| Markers | Annotations `<type>/<label>` (`state/left`, `cue/right`, `artifact/blink`) at the sample they apply to |
| Stream gaps | `BAD_gap` annotations where samples are missing |
| Subject | `subject_info.his_id` = the pseudonym only |
| Stream metadata | JSON in the file description: model, driver, reference, montage, `spec_version`, software version |

## Replaying a recording

```bash
openmind-plug stream --source replay --file ~/openmind-data/recordings/P01/P01_20261118-140509_raw.fif
```

`ReplaySource.from_file()` restores the model, reference and montage, and
re-emits the markers with their original type and label (`"source": "replay"`).
Samples are identical to those recorded.

XDF export is not supported yet.
