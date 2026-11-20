# Signal quality

`openmind-plug quality` checks a live stream channel by channel, and runs the
eyes open / eyes closed **alpha test** that confirms a headset really records
brain activity.

## Live checks

```bash
openmind-plug quality              # refreshes every 2 s, Ctrl+C to stop
openmind-plug quality --line-freq 60 --saturation-uv 187500
```

Every window (`--window`, 2 s by default) each channel gets one status. All
checks run on data high-passed at 1 Hz, so DC offsets and slow drifts do not
hide real problems.

| Status | Meaning | Rule | Usual cause |
|---|---|---|---|
| `good` | No problem found | | |
| `flat` | Almost no signal | standard deviation < 0.5 µV | electrode disconnected, channel at zero |
| `saturated` | Amplifier clipping | > 5 % of samples at the window extremes, or beyond `--saturation-uv` | large offset, broken reference |
| `line noise` | Strong 50/60 Hz peak | peak > 10 × its spectral neighbourhood (10 dB) | poor contact, unshielded cable |
| `noisy` | Much noisier than the others | > 5 × the median standard deviation | muscle, movement, loose electrode |
| `weak` | Much quieter than the others | < 1/5 of the median standard deviation | partial contact, bridged electrodes |

When several problems apply, the first in this table wins. `noisy` and `weak`
need at least three active channels. Drivers that measure electrode impedance
report it through `EEGSource.impedances()`; `assess()` includes it in each
`ChannelQuality`.

## Alpha test

```bash
openmind-plug quality --alpha-test           # 30 s eyes open, then 30 s eyes closed
```

Closing the eyes raises the alpha rhythm (8-12 Hz) over the back of the head
(Berger effect). The test compares alpha power between the two phases on every
channel and **passes when the best channel rises at least ×2**. With the synthetic
source, press `e` in the `stream` terminal when the second phase starts.

**Montages without posterior electrodes.** The motor montages `A` and `C` have
no occipital or parietal electrode. On a real head, alpha also spreads to the
central-parietal sites (`CP3`, `CP4`, `CPz`), but its rise there is weaker and may
stay below ×2; the synthetic source models alpha over the back of the head only,
so with `A` or `C` its test does not pass (about ×1.2 to ×1.3). To check a
headset, run the test with montage `B` (`O1`, `O2`), or move one electrode to
`Pz` or `O1` for the test.

Phases shorter than 20 s give unreliable results: on the best of 8 channels,
chance alone can reach ×1.5 with 30 s phases and ×3 with 6 s phases.

## From Python

```python
from openmind_plug.quality import alpha_test, assess

report = assess(samples, srate=250, labels=info.labels, line_freq=50)
bad = [channel.label for channel in report if not channel.ok]

result = alpha_test(eyes_open, eyes_closed, srate=250, labels=info.labels)
print(result.passed, result.best_channel, result.best_gain)
```
