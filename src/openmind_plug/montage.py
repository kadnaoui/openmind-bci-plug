"""Electrode positions (10-10 system) and named montages.

Positions are approximate 3D coordinates in meters, in the MNE head frame
(x towards the right ear, y towards the nose, z towards the vertex), taken from the
MNE ``standard_1005`` template. They are meant for visualization and spatial
filters, not for source localization.

Named montages describe which electrode is wired to each amplifier input, in order:

- ``A`` (default): motor imagery, centred on the sensorimotor cortex.
- ``B``: SSVEP variant, trades the frontal-central pair for occipital electrodes.
"""

from __future__ import annotations

from dataclasses import dataclass

from openmind_plug.sources.base import ChannelInfo

#: Position of an electrode, ``(x, y, z)`` in meters.
Position = tuple[float, float, float]

POSITIONS_10_10: dict[str, Position] = {
    "Fp1": (-0.0294, 0.0839, -0.0070),
    "Fp2": (0.0299, 0.0849, -0.0071),
    "AF3": (-0.0337, 0.0768, 0.0212),
    "AF4": (0.0357, 0.0777, 0.0220),
    "F7": (-0.0703, 0.0425, -0.0114),
    "F3": (-0.0502, 0.0531, 0.0422),
    "Fz": (0.0003, 0.0585, 0.0665),
    "F4": (0.0518, 0.0543, 0.0408),
    "F8": (0.0730, 0.0444, -0.0120),
    "FC5": (-0.0772, 0.0186, 0.0245),
    "FC3": (-0.0602, 0.0227, 0.0555),
    "FC1": (-0.0341, 0.0260, 0.0800),
    "FCz": (0.0004, 0.0274, 0.0887),
    "FC2": (0.0348, 0.0264, 0.0788),
    "FC4": (0.0623, 0.0237, 0.0556),
    "FC6": (0.0795, 0.0199, 0.0244),
    "T7": (-0.0842, -0.0160, -0.0093),
    "C5": (-0.0803, -0.0138, 0.0292),
    "C3": (-0.0654, -0.0116, 0.0644),
    "C1": (-0.0362, -0.0100, 0.0898),
    "Cz": (0.0004, -0.0092, 0.1002),
    "C2": (0.0377, -0.0096, 0.0884),
    "C4": (0.0671, -0.0109, 0.0636),
    "C6": (0.0835, -0.0128, 0.0292),
    "T8": (0.0851, -0.0150, -0.0095),
    "CP5": (-0.0796, -0.0466, 0.0309),
    "CP3": (-0.0636, -0.0470, 0.0656),
    "CP1": (-0.0355, -0.0473, 0.0913),
    "CPz": (0.0004, -0.0473, 0.0994),
    "CP2": (0.0384, -0.0471, 0.0907),
    "CP4": (0.0666, -0.0466, 0.0656),
    "CP6": (0.0833, -0.0461, 0.0312),
    "P7": (-0.0724, -0.0735, -0.0025),
    "P3": (-0.0530, -0.0788, 0.0559),
    "P1": (-0.0286, -0.0805, 0.0754),
    "Pz": (0.0003, -0.0811, 0.0826),
    "P2": (0.0319, -0.0805, 0.0767),
    "P4": (0.0557, -0.0786, 0.0566),
    "P8": (0.0731, -0.0731, -0.0025),
    "PO7": (-0.0548, -0.0975, 0.0028),
    "PO3": (-0.0365, -0.1009, 0.0372),
    "POz": (0.0002, -0.1022, 0.0506),
    "PO4": (0.0368, -0.1008, 0.0364),
    "PO8": (0.0557, -0.0976, 0.0027),
    "O1": (-0.0294, -0.1124, 0.0088),
    "Oz": (0.0001, -0.1149, 0.0147),
    "O2": (0.0298, -0.1122, 0.0088),
}

_CANONICAL_LABELS: dict[str, str] = {label.casefold(): label for label in POSITIONS_10_10}


def canonical_label(label: str) -> str:
    """Return the canonical spelling of a 10-10 label (``"CZ"`` → ``"Cz"``).

    Raises:
        KeyError: If the label is not a known 10-10 position.
    """
    try:
        return _CANONICAL_LABELS[label.strip().casefold()]
    except KeyError:
        raise KeyError(f"unknown 10-10 electrode label {label!r}") from None


def position(label: str) -> Position:
    """Return the ``(x, y, z)`` position in meters of a 10-10 electrode.

    The lookup is case-insensitive.

    Raises:
        KeyError: If the label is not a known 10-10 position.
    """
    return POSITIONS_10_10[canonical_label(label)]


@dataclass(frozen=True)
class Montage:
    """Ordered set of electrodes wired to the amplifier inputs.

    Args:
        name: Short montage name (``"A"``).
        labels: Electrode labels, in amplifier input order; each must be a known,
            unique 10-10 position.
        description: One-line description of the intended use.
    """

    name: str
    labels: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("montage name must not be empty")
        if not self.labels:
            raise ValueError(f"montage {self.name!r} has no electrodes")
        unknown = [label for label in self.labels if label not in POSITIONS_10_10]
        if unknown:
            raise ValueError(f"montage {self.name!r} has unknown electrodes: {unknown}")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"montage {self.name!r} has duplicate electrodes")

    @property
    def n_channels(self) -> int:
        """Number of electrodes."""
        return len(self.labels)

    def channels(self, unit: str = "uV") -> tuple[ChannelInfo, ...]:
        """Return the montage as EEG :class:`ChannelInfo` objects, in input order."""
        return tuple(ChannelInfo(label=label, unit=unit, type="EEG") for label in self.labels)

    def positions(self) -> dict[str, Position]:
        """Return the position of each electrode, in input order."""
        return {label: POSITIONS_10_10[label] for label in self.labels}


MONTAGE_A = Montage(
    name="A",
    labels=("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Fp1"),
    description="Motor imagery: sensorimotor cortex, plus Fp1 for blink detection.",
)

MONTAGE_B = Montage(
    name="B",
    labels=("C3", "Cz", "C4", "CP3", "CP4", "Fp1", "O1", "O2"),
    description="SSVEP variant: occipital O1/O2 instead of FC3/FC4.",
)

#: Named montages, by name.
MONTAGES: dict[str, Montage] = {m.name: m for m in (MONTAGE_A, MONTAGE_B)}

#: Name of the montage used when none is given.
DEFAULT_MONTAGE = "A"


def get_montage(name: str = DEFAULT_MONTAGE) -> Montage:
    """Return a named montage (case-insensitive).

    Raises:
        KeyError: If no montage has this name; the message lists the available ones.
    """
    for montage_name, montage in MONTAGES.items():
        if montage_name.casefold() == name.strip().casefold():
            return montage
    raise KeyError(f"unknown montage {name!r}; available montages: {sorted(MONTAGES)}")
