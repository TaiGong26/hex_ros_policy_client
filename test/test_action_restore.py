################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

import numpy as np
import pytest

from hex_util_ros import quat_inv, quat_mul

from hex_ros_policy_client.process.action.action_restore import (
    ACTION_RESTORERS,
    as_received,
    ee_delta,
    pos_delta,
    pos_diff,
    received_to_absolute,
)

# One reference segment of the model side: an end effector trajectory of 15
# steps for two arms, as fourteens wide rows. The left arm is the first seven
# columns and the right arm is the last seven. The segment was published with
# its anchor, its absolute poses and its increments, so a restorer can be
# compared against numbers it did not produce.

ANCHOR_LEFT = np.array([
    [0.365388303995, -0.184985443950, 0.244562715292,
     0.874076187611, -0.087940670550, 0.272740304470, -0.392262607813],
])

ANCHOR_RIGHT = np.array([
    [0.358207195997, 0.202524527907, 0.259021729231,
     0.853082180023, 0.067363262177, 0.295854538679, 0.424479812384],
])

ABS_LEFT = np.array([
    [0.357348710299, -0.170207425952, 0.272431522608,
     0.896930396557, -0.061211213470, 0.222728729248, -0.377042442560],
    [0.349497228861, -0.170284748077, 0.270245283842,
     0.897483348846, -0.056107249111, 0.219884693623, -0.378188252449],
    [0.342563807964, -0.170364752412, 0.268019706011,
     0.897998809814, -0.051782529801, 0.217218786478, -0.379121005535],
    [0.336542546749, -0.170457035303, 0.266504079103,
     0.898303747177, -0.047820597887, 0.214667454362, -0.380370140076],
    [0.331576257944, -0.170299082994, 0.265057712793,
     0.898931920528, -0.043661132455, 0.211969792843, -0.380898773670],
    [0.327434897423, -0.169909819961, 0.263749331236,
     0.899486362934, -0.040664281696, 0.210297957063, -0.380848407745],
    [0.323127031326, -0.170039787889, 0.261692285538,
     0.899143695831, -0.037725262344, 0.209673941135, -0.382301270962],
    [0.319349974394, -0.170421957970, 0.259721606970,
     0.898721873760, -0.034775026143, 0.209454834461, -0.383690416813],
    [0.315910667181, -0.171230345964, 0.257642149925,
     0.897682189941, -0.032409489155, 0.209751218557, -0.386161565781],
    [0.312053322792, -0.172464936972, 0.255673736334,
     0.896260678768, -0.029781924561, 0.210465073586, -0.389274060726],
    [0.308660060167, -0.173797875643, 0.253516286612,
     0.894732296467, -0.026059962809, 0.210650727153, -0.392939209938],
    [0.306407272816, -0.174188852310, 0.251295059919,
     0.894011855125, -0.023996284232, 0.211608916521, -0.394193649292],
    [0.304415792227, -0.174844607711, 0.248157739639,
     0.892603278160, -0.023476647213, 0.214396640658, -0.395906895399],
    [0.302920162678, -0.175039961934, 0.244905516505,
     0.891411662102, -0.022597629577, 0.217235699296, -0.397093504667],
    [0.302851676941, -0.174085691571, 0.241681933403,
     0.891052424908, -0.023219734430, 0.220765754580, -0.395915269852],
])

ABS_RIGHT = np.array([
    [0.347366422415, 0.190682068467, 0.307809650898,
     0.880312204361, 0.035813666880, 0.224674507976, 0.416280180216],
    [0.339233934879, 0.192338645458, 0.304440110922,
     0.878763616085, 0.032793030143, 0.224820688367, 0.419707924128],
    [0.332448989153, 0.193797960877, 0.301249474287,
     0.877578973770, 0.029510319233, 0.224659457803, 0.422507345676],
    [0.325125873089, 0.195425346494, 0.297860264778,
     0.875803589821, 0.026858255267, 0.225121095777, 0.426107078791],
    [0.318466484547, 0.197182729840, 0.294222027063,
     0.873580932617, 0.024691328406, 0.226768270135, 0.429910242558],
    [0.312515258789, 0.199083149433, 0.291198551655,
     0.871399402618, 0.022147295997, 0.227431029081, 0.434105575085],
    [0.307000964880, 0.200278222561, 0.289746522903,
     0.870144844055, 0.018677890301, 0.225849747658, 0.437596857548],
    [0.302013754845, 0.201486140490, 0.287567049265,
     0.868579626083, 0.015746822581, 0.225471928716, 0.441003292799],
    [0.296703398228, 0.203359946609, 0.286148667336,
     0.866485416889, 0.013245638460, 0.224658608437, 0.445596367121],
    [0.291447252035, 0.205435216427, 0.284279704094,
     0.863503038883, 0.010343939066, 0.225259810686, 0.451124697924],
    [0.286391645670, 0.207528278232, 0.282939046621,
     0.860408425331, 0.007333149202, 0.225881651044, 0.456750571728],
    [0.281180769205, 0.209919616580, 0.281300246716,
     0.857107460499, 0.004288773518, 0.226120054722, 0.462837010622],
    [0.277391046286, 0.211329907179, 0.278897047043,
     0.854542076588, 0.001636602450, 0.227243497968, 0.467028439045],
    [0.273767441511, 0.212905988097, 0.276975333691,
     0.852150022984, -0.000422192214, 0.228339686990, 0.470851510763],
    [0.272026747465, 0.213328853250, 0.274975061417,
     0.851093292236, -0.002683338476, 0.229618191719, 0.472131818533],
])

DELTA_LEFT = np.array([
    [-0.008039593697, 0.014778017998, 0.027868807316,
     0.998015209936, 0.009906877989, -0.059093685914, 0.019376286817],
    [-0.007851481438, -0.000077322125, -0.002186238766,
     0.999982149629, 0.003284220737, -0.004668607964, -0.001781937860],
    [-0.006933420897, -0.000080004334, -0.002225577831,
     0.999986551255, 0.002696971775, -0.004193841497, -0.001443551945],
    [-0.006021261215, -0.000092282891, -0.001515626907,
     0.999988081266, 0.002335001485, -0.003924066285, -0.001734604782],
    [-0.004966288805, 0.000157952309, -0.001446366310,
     0.999987349853, 0.002626912691, -0.004165583849, -0.000999833222],
    [-0.004141360521, 0.000389263034, -0.001308381557,
     0.999993958169, 0.002092048248, -0.002759689336, -0.000305785621],
    [-0.004307866096, -0.000129967928, -0.002057045698,
     0.999994395208, 0.002086483877, -0.001667632105, -0.002030029477],
    [-0.003777056932, -0.000382170081, -0.001970678568,
     0.999994558716, 0.002261740488, -0.001288848230, -0.002020626643],
    [-0.003439307213, -0.000808387995, -0.002079457045,
     0.999993559510, 0.001685930442, -0.000509434470, -0.003125572271],
    [-0.003857344389, -0.001234591007, -0.001968413591,
     0.999990452292, 0.001935461256, -0.000176560183, -0.003917234812],
    [-0.003393262625, -0.001332938671, -0.002157449722,
     0.999985153856, 0.002591214019, -0.001069953276, -0.004668760780],
    [-0.002252787352, -0.000390976667, -0.002221226692,
     0.999996335732, 0.001939926810, 0.000165493594, -0.001865162828],
    [-0.001991480589, -0.000655755401, -0.003137320280,
     0.999993489611, 0.001167126040, 0.002544376720, -0.002263769272],
    [-0.001495629549, -0.000195354223, -0.003252223134,
     0.999994149136, 0.001626236724, 0.002413764776, -0.001786050343],
    [-0.000068485737, 0.000954270363, -0.003223583102,
     0.999992819198, 0.001095047258, 0.003498430702, 0.000963014000],
])

DELTA_RIGHT = np.array([
    [-0.010840773582, -0.011842459440, 0.048787921667,
     0.996564685536, -0.000960113117, -0.081618354376, -0.014014383815],
    [-0.008132487535, 0.001656576991, -0.003369539976,
     0.999988342481, -0.001894368083, -0.000903578593, 0.004346026734],
    [-0.006784945726, 0.001459315419, -0.003190636635,
     0.999989980112, -0.002148841274, -0.001344932750, 0.003689967658],
    [-0.007323116064, 0.001627385616, -0.003389209509,
     0.999988365417, -0.001661334772, -0.000422765205, 0.004518597189],
    [-0.006659388542, 0.001757383347, -0.003638237715,
     0.999986601845, -0.001683805889, 0.000917479023, 0.004809975627],
    [-0.005951225758, 0.001900419593, -0.003023475409,
     0.999985323100, -0.001502111891, -0.000123618561, 0.005196094820],
    [-0.005514293909, 0.001195073128, -0.001452028751,
     0.999985838095, -0.001514983926, -0.002676012964, 0.004340941719],
    [-0.004987210035, 0.001207917929, -0.002179473639,
     0.999988634014, -0.001586543571, -0.001321504168, 0.004303950580],
    [-0.005310356617, 0.001873806119, -0.001418381929,
     0.999983831033, -0.000745214181, -0.001409604797, 0.005464143525],
    [-0.005256146193, 0.002075269818, -0.001868963242,
     0.999975884828, -0.001500683141, -0.000175263105, 0.006779009881],
    [-0.005055606365, 0.002093061805, -0.001340657473,
     0.999974665251, -0.001581059978, -0.000182382262, 0.006938657964],
    [-0.005210876465, 0.002391338348, -0.001638799906,
     0.999971371157, -0.001329276171, -0.000484401179, 0.007433957735],
    [-0.003789722919, 0.001410290599, -0.002403199673,
     0.999983766969, -0.001834398347, 0.000297497322, 0.005384386453],
    [-0.003623604774, 0.001576080918, -0.001921713352,
     0.999987114534, -0.001398595084, 0.000512545765, 0.004853774433],
    [-0.001740694046, 0.000422865152, -0.002000272274,
     0.999995227485, -0.002236922791, 0.000266647805, 0.002104347219],
])


ARMS = (
    (ANCHOR_LEFT[0], ABS_LEFT, DELTA_LEFT),
    (ANCHOR_RIGHT[0], ABS_RIGHT, DELTA_RIGHT),
)


def forward_ee(absolute: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Build the increments of one reference segment, one arm at a time.

    Args:
        absolute (np.ndarray): Absolute poses of shape ``(steps, 7)``.
        anchor (np.ndarray): Pose of the chunk start, shape ``(7,)``.

    Returns:
        np.ndarray: Increments of shape ``(steps, 7)``.
    """
    increments = np.empty_like(absolute)
    increments[0, :3] = absolute[0, :3] - anchor[:3]
    increments[0, 3:] = quat_mul(absolute[0, 3:], quat_inv(anchor[3:]))
    if len(absolute) > 1:
        increments[1:, :3] = absolute[1:, :3] - absolute[:-1, :3]
        increments[1:, 3:] = quat_mul(absolute[1:, 3:],
                                      quat_inv(absolute[:-1, 3:]))
    return increments


@pytest.mark.parametrize("anchor, absolute, delta", ARMS)
def test_ee_delta_restores_the_reference_segment(
        anchor: np.ndarray, absolute: np.ndarray,
        delta: np.ndarray) -> None:
    """The increments of the reference segment give back its poses."""
    assert np.allclose(ee_delta(delta, anchor), absolute, atol=1e-7)


@pytest.mark.parametrize("anchor, absolute, delta", ARMS)
def test_ee_round_trip_keeps_the_poses(anchor: np.ndarray,
                                       absolute: np.ndarray,
                                       delta: np.ndarray) -> None:
    """Increments built from the poses restore to those same poses."""
    increments = forward_ee(absolute, anchor)
    assert np.allclose(ee_delta(increments, anchor), absolute, atol=1e-9)


@pytest.mark.parametrize("anchor, absolute, delta", ARMS)
def test_ee_delta_returns_unit_quaternions_with_positive_w(
        anchor: np.ndarray, absolute: np.ndarray,
        delta: np.ndarray) -> None:
    """Every restored orientation is normalized and sits in one hemisphere."""
    restored = ee_delta(delta, anchor)
    quaternions = restored[:, 3:]
    assert np.allclose(np.linalg.norm(quaternions, axis=1), 1.0)
    assert bool((quaternions[:, 0] >= 0.0).all())


def test_ee_delta_without_increments_gives_the_anchor() -> None:
    """An anchor with no movement stays where it is."""
    still = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    assert np.allclose(ee_delta(still, ANCHOR_LEFT[0]), ANCHOR_LEFT[0])


def test_ee_delta_rejects_a_zero_quaternion() -> None:
    """A zero quaternion names no rotation and must not become a NaN pose."""
    steps = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        ee_delta(steps, ANCHOR_LEFT[0])


def test_ee_delta_needs_the_anchor() -> None:
    """A pose increment cannot be restored without the pose it starts from."""
    with pytest.raises(ValueError):
        ee_delta(DELTA_LEFT)


def test_ee_delta_rejects_a_row_that_is_not_a_pose() -> None:
    """A row of six values cannot carry a position and a quaternion."""
    with pytest.raises(ValueError):
        ee_delta(np.zeros((2, 6)), ANCHOR_LEFT[0])


def test_ee_delta_rejects_an_anchor_that_is_not_a_pose() -> None:
    """The anchor has to hold a position and a quaternion too."""
    with pytest.raises(ValueError):
        ee_delta(DELTA_LEFT, np.zeros(6))


def test_as_received_keeps_the_values() -> None:
    """Values that are already absolute pass through untouched."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert as_received(matrix) is matrix
    assert np.array_equal(as_received(ABS_LEFT), ABS_LEFT)


def test_pos_delta_adds_the_run_to_the_anchor() -> None:
    """Per step increments become absolute values from the anchor."""
    restored = pos_delta(np.array([[1.0], [1.0], [2.0]]), np.array([10.0]))
    assert restored[:, 0].tolist() == [11.0, 12.0, 14.0]


def test_pos_delta_restores_the_reference_segment() -> None:
    """The joint segment of the reference document is reproduced."""
    delta = np.array([[0.029287163168, -0.111202389002, 0.020578861237],
                      [-0.001533979550, -0.030584216118, 0.024543762207]])
    reference = np.array([[-0.027611654252, -0.531468391418, 2.233475923538],
                          [-0.029145633802, -0.562052607536, 2.258019685745]])
    anchor = np.array([-0.056898817420, -0.420266002417, 2.212897062302])
    assert np.allclose(pos_delta(delta, anchor), reference, atol=1e-11)


def test_pos_delta_round_trip_keeps_the_positions() -> None:
    """Increments built by differencing restore to the same positions."""
    absolute = np.array([[0.0, 1.0], [0.5, 1.5], [2.0, 0.5], [2.5, -0.5]])
    anchor = np.array([0.0, 1.0])
    increments = np.diff(np.vstack([anchor, absolute]), axis=0)
    assert np.allclose(pos_delta(increments, anchor), absolute)


def test_pos_delta_handles_a_single_step() -> None:
    """A chunk of one row restores its only row."""
    assert np.allclose(pos_delta(np.array([[0.25]]), np.array([1.0])),
                       np.array([[1.25]]))


def test_pos_diff_adds_the_anchor_to_every_row() -> None:
    """Per row differences become absolute values from the anchor."""
    restored = pos_diff(np.array([[1.0], [1.0], [2.0]]), np.array([10.0]))
    assert restored[:, 0].tolist() == [11.0, 11.0, 12.0]


def test_pos_diff_round_trip_keeps_the_positions() -> None:
    """Rows built by subtracting the anchor restore to the same positions."""
    absolute = np.array([[0.0, 1.0], [0.5, 1.5], [2.0, 0.5]])
    anchor = np.array([0.0, 1.0])
    assert np.allclose(pos_diff(absolute - anchor, anchor), absolute)


def test_pos_delta_needs_the_anchor() -> None:
    """An increment cannot be restored without the value it starts from."""
    with pytest.raises(ValueError):
        pos_delta(np.array([[1.0]]))


def test_pos_diff_needs_the_anchor() -> None:
    """A difference cannot be restored without the value it refers to."""
    with pytest.raises(ValueError):
        pos_diff(np.array([[1.0]]))


def test_every_supported_pair_has_a_restorer() -> None:
    """The table holds exactly the pairs this package supports."""
    expected = {
        ("pos", "abs"): "as_received",
        ("pos", "delta"): "pos_delta",
        ("pos", "diff"): "pos_diff",
        ("ee", "abs"): "as_received",
        ("ee", "delta"): "ee_delta",
    }
    found = {key: fn.__name__ for key, fn in ACTION_RESTORERS.items()}
    assert found == expected


def test_received_to_absolute_needs_no_anchor_for_absolute_values() -> None:
    """The abs mode needs no anchor and gives the values back."""
    matrix = np.zeros((2, 7))
    assert received_to_absolute("pos", "abs", matrix) is matrix
    assert received_to_absolute("ee", "abs", matrix) is matrix


def test_received_to_absolute_dispatches_per_step_increments() -> None:
    """The delta mode adds the running sum through the right restorer."""
    restored = received_to_absolute("pos", "delta", np.array([[1.0], [1.0]]),
                       np.array([0.0]))
    assert restored[:, 0].tolist() == [1.0, 2.0]


def test_received_to_absolute_dispatches_per_row_differences() -> None:
    """The diff mode adds the anchor through the right restorer."""
    restored = received_to_absolute("pos", "diff", np.array([[1.0], [1.0]]),
                       np.array([0.0]))
    assert restored[:, 0].tolist() == [1.0, 1.0]


def test_end_effector_diff_is_not_supported() -> None:
    """A pose difference has no restorer yet and is refused, not guessed."""
    assert ("ee", "diff") not in ACTION_RESTORERS
    with pytest.raises(ValueError):
        received_to_absolute("ee", "diff", ABS_LEFT, ANCHOR_LEFT[0])


def test_unknown_mode_is_rejected() -> None:
    """A mode outside the tables raises instead of passing through."""
    matrix = np.zeros((2, 7))
    with pytest.raises(ValueError):
        received_to_absolute("pos", "unchanged", matrix, np.zeros(7))
    with pytest.raises(ValueError):
        received_to_absolute("wheel", "delta", matrix, np.zeros(7))


def test_unhashable_mode_is_a_value_error() -> None:
    """A mode that cannot key the table is refused like any other bad mode."""
    with pytest.raises(ValueError):
        received_to_absolute(["pos"], "delta", np.zeros((2, 7)), np.zeros(7))


def test_restorers_keep_the_shape() -> None:
    """Every restorer returns the shape it was given."""
    for (arm_ctrl_mode, _), restorer in ACTION_RESTORERS.items():
        if arm_ctrl_mode == "ee":
            matrix = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                             (4, 1))
            anchor = matrix[0]
        else:
            matrix = np.zeros((4, 7))
            anchor = np.zeros(7)
        assert restorer(matrix, anchor).shape == matrix.shape
