import math
import matplotlib.pyplot as plt
import numpy as np
import pytest
from kneed.data_generator import DataGenerator as dg
from kneed.knee_locator import KneeLocator
from kneed.shape_detector import find_shape

@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_figure2(interp_method):
    """From the kneedle manuscript"""
    x, y = dg.figure2()
    kl = KneeLocator(
        x,
        y,
        S=1.0,
        curve="concave",
        interp_method=interp_method,
    )
    assert math.isclose(kl.knee, 0.22, rel_tol=0.05)
    assert math.isclose(kl.elbow, 0.22, rel_tol=0.05)
    assert math.isclose(kl.norm_elbow, kl.knee, rel_tol=0.05)


def test_NoisyGaussian():
    """From the Kneedle manuscript"""
    x, y = dg.noisy_gaussian(mu=50, sigma=10, N=1000, seed=42)
    kl = KneeLocator(
        x,
        y,
        S=1.0,
        curve="concave",
        interp_method="polynomial",
        polynomial_degree=11,
        online=True,
    )
    assert math.isclose(kl.knee, 63.0, rel_tol=1e-02)


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_concave_increasing(interp_method):
    """test a concave increasing function"""
    x, y = dg().concave_increasing()
    kn = KneeLocator(
        x,
        y,
        curve="concave",
        interp_method=interp_method,
    )
    assert kn.knee == 2


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_concave_decreasing(interp_method):
    """test a concave decreasing function"""
    x, y = dg.concave_decreasing()
    kn = KneeLocator(
        x,
        y,
        curve="concave",
        direction="decreasing",
        interp_method=interp_method,
    )
    assert kn.knee == 7


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_convex_increasing(interp_method):
    """test a convex increasing function"""
    x, y = dg.convex_increasing()
    kl = KneeLocator(
        x,
        y,
        curve="convex",
        interp_method=interp_method,
    )
    assert kl.knee == 7


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_convex_decreasing(interp_method):
    """test a convex decreasing function"""
    x, y = dg.convex_decreasing()
    kl = KneeLocator(
        x,
        y,
        curve="convex",
        direction="decreasing",
        interp_method=interp_method,
    )
    assert kl.knee == 2


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_concave_increasing_truncated(interp_method):
    """test a truncated concave increasing function"""
    x, y = dg.concave_increasing()
    kl = KneeLocator(
        x[:-3] / 10,
        y[:-3] / 10,
        curve="concave",
        interp_method=interp_method,
    )
    assert kl.knee == 0.2


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_concave_decreasing_truncated(interp_method):
    """test a truncated concave decreasing function"""
    x, y = dg.concave_decreasing()
    kl = KneeLocator(
        x[:-3] / 10,
        y[:-3] / 10,
        curve="concave",
        direction="decreasing",
        interp_method=interp_method,
    )
    assert kl.knee == 0.4


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_convex_increasing_truncated(interp_method):
    """test a truncated convex increasing function"""
    x, y = dg.convex_increasing()
    kl = KneeLocator(
        x[:-3] / 10,
        y[:-3] / 10,
        curve="convex",
        interp_method=interp_method,
    )
    assert kl.knee == 0.4


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_convex_decreasing_truncated(interp_method):
    """test a truncated convex decreasing function"""
    x, y = dg.convex_decreasing()
    kl = KneeLocator(
        x[:-3] / 10,
        y[:-3] / 10,
        curve="convex",
        direction="decreasing",
        interp_method=interp_method,
    )
    assert kl.knee == 0.2


@pytest.mark.parametrize(
    "interp_method, expected",
    [("interp1d", 26), ("polynomial", 28), ("make_splrep", 26)],
)
def test_convex_decreasing_bumpy(interp_method, expected):
    """test a bumpy convex decreasing function"""
    x, y = dg.bumpy()
    kl = KneeLocator(
        x,
        y,
        curve="convex",
        direction="decreasing",
        interp_method=interp_method,
    )
    assert kl.knee == expected


@pytest.mark.parametrize("interp_method", ["interp1d", "make_splrep"])
@pytest.mark.parametrize("online, expected", [(True, 482), (False, 22)])
def test_gamma_online_offline(online, expected, interp_method):
    """
    Tests online and offline knee detection for the default and spline methods.
    Notable that a large number of samples are highly sensitive to S parameter.
    """
    np.random.seed(23)
    n = 1000
    x = range(1, n + 1)
    y = sorted(np.random.gamma(0.5, 1.0, n), reverse=True)
    kl = KneeLocator(
        x,
        y,
        curve="convex",
        direction="decreasing",
        online=online,
        interp_method=interp_method,
    )
    assert kl.knee == expected


def test_sensitivity():
    """Test the S parameter -- where S is the number of flat points to identify before calling a knee"""
    np.random.seed(23)
    sensitivity = [1, 3, 5, 10, 100, 200, 400]
    detected_knees = []
    expected_knees = [43, 137, 178, 258, 305, 482, 482]
    n = 1000
    x = range(1, n + 1)
    y = sorted(np.random.gamma(0.5, 1.0, n), reverse=True)
    for s, expected_knee in zip(sensitivity, expected_knees):
        kl = KneeLocator(x, y, curve="convex", direction="decreasing", S=s)
        detected_knees.append(kl.knee)
        assert kl.knee, expected_knee


@pytest.mark.parametrize("interp_method", ["interp1d", "make_splrep"])
def test_sine(interp_method):
    x = np.arange(0, 10, 0.1)
    y_sin = np.sin(x)

    sine_combos = [
        ("decreasing", "convex"),
        ("increasing", "convex"),
        ("increasing", "concave"),
        ("decreasing", "concave"),
    ]
    expected_knees = [4.5, 4.9, 7.7, 1.8]
    detected_knees = []
    for direction, curve in sine_combos:
        kl_sine = KneeLocator(
            x,
            y_sin,
            direction=direction,
            curve=curve,
            S=1,
            online=True,
            interp_method=interp_method,
        )
        detected_knees.append(kl_sine.knee)
    assert np.isclose(expected_knees, detected_knees).all()


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial", "make_splrep"])
def test_list_input(interp_method):
    """Indirectly test that flip works on lists as input"""
    x, y = dg.figure2()
    kl = KneeLocator(
        x.tolist(),
        y.tolist(),
        S=1.0,
        curve="concave",
        interp_method=interp_method,
    )
    assert math.isclose(kl.knee, 0.22, rel_tol=0.05)


@pytest.mark.parametrize("interp_method", ["interp1d", "make_splrep"])
def test_flat_maxima(interp_method):
    """The global maxima has a sequentially equal value in the difference curve."""
    x = [
        0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
    ]
    y = [
        1,
        0.787701317715959,
        0.7437774524158126,
        0.6559297218155198,
        0.5065885797950219,
        0.36749633967789164,
        0.2547584187408492,
        0.16251830161054173,
        0.10395314787701318,
        0.06734992679355783,
        0.043923865300146414,
        0.027818448023426062,
        0.01903367496339678,
        0.013177159590043924,
        0.010248901903367497,
        0.007320644216691069,
        0.005856515373352855,
        0.004392386530014641,
    ]
    # When S=0.0 the first local maximum is found.
    kl = KneeLocator(
        x,
        y,
        curve="convex",
        direction="decreasing",
        S=0.0,
        interp_method=interp_method,
    )
    assert math.isclose(kl.knee, 1.0, rel_tol=0.05)

    # When S=1.0 the global maximum is found.
    kl = KneeLocator(
        x,
        y,
        curve="convex",
        direction="decreasing",
        S=1.0,
        interp_method=interp_method,
    )
    assert math.isclose(kl.knee, 8.0, rel_tol=0.05)


def test_all_knees():
    x, y = dg.bumpy()
    kl = KneeLocator(x, y, curve="convex", direction="decreasing", online=True)
    assert np.isclose(sorted(kl.all_elbows), [26, 31, 41, 46, 53]).all()
    assert np.isclose(
        sorted(kl.all_norm_elbows),
        [
            0.2921348314606742,
            0.348314606741573,
            0.4606741573033708,
            0.5168539325842696,
            0.5955056179775281,
        ],
    ).all()


def test_y():
    """Test the y value"""
    x, y = dg.figure2()
    kl = KneeLocator(x, y, S=1.0, curve="concave", interp_method="interp1d")
    assert math.isclose(kl.knee_y, 1.897, rel_tol=0.03)
    assert math.isclose(kl.all_knees_y[0], 1.897, rel_tol=0.03)
    assert math.isclose(kl.norm_knee_y, 0.758, rel_tol=0.03)
    assert math.isclose(kl.all_norm_knees_y[0], 0.758, rel_tol=0.03)

    assert math.isclose(kl.elbow_y, 1.897, rel_tol=0.03)
    assert math.isclose(kl.all_elbows_y[0], 1.897, rel_tol=0.03)
    assert math.isclose(kl.norm_elbow_y, 0.758, rel_tol=0.03)
    assert math.isclose(kl.all_norm_elbows_y[0], 0.758, rel_tol=0.03)


def test_y_no_knee():
    """Test the y value, if there is no knee found."""
    kl = KneeLocator(
        np.array([1, 2, 3]),
        np.array([0.90483742, 0.81873075, 0.74081822]),
        S=1.0,
        curve="convex",
        direction="decreasing",
        interp_method="interp1d",
        online=False,
    )
    assert kl.knee_y is None
    assert kl.norm_knee_y is None


def test_interp_method():
    """Test that the interp_method argument is valid."""
    x, y = dg.figure2()
    with pytest.raises(ValueError):
        kl = KneeLocator(x, y, interp_method="not_a_method")


def test_make_splrep_requires_smoothing_factor():
    """make_splrep requires a smoothing_factor value."""
    x, y = dg.figure2()
    with pytest.raises(ValueError, match="smoothing_factor"):
        KneeLocator(
            x,
            y,
            S=1.0,
            curve="concave",
            interp_method="make_splrep",
            smoothing_factor=None,
        )


def test_x_equals_y():
    """Test that knee is None when no maxima are found"""
    x = range(10)
    y = [1] * len(x)
    kl = KneeLocator(x, y)
    assert kl.knee is None



def test_plot_knee_normalized():
    """Test that plotting is functional"""
    x, y = dg.figure2()
    kl = KneeLocator(x, y, S=1.0, curve="concave", interp_method="interp1d")
    num_figures_before = plt.gcf().number
    kl.plot_knee_normalized()
    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after


def test_plot_knee():
    """Test that plotting is functional"""
    x, y = dg.figure2()
    kl = KneeLocator(x, y, S=1.0, curve="concave", interp_method="interp1d")
    num_figures_before = plt.gcf().number
    kl.plot_knee()
    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after


@pytest.mark.parametrize("interp_method", ["interp1d", "make_splrep"])
def test_logistic(interp_method):
    y = np.array(
        [
            2.00855493e-45,
            1.10299045e-43,
            4.48168384e-42,
            1.22376580e-41,
            5.10688883e-40,
            1.18778110e-38,
            5.88777891e-35,
            4.25317895e-34,
            4.06507035e-33,
            6.88084518e-32,
            2.99321831e-31,
            1.13291723e-30,
            1.05244482e-28,
            2.67578448e-27,
            1.22522190e-26,
            2.36517846e-26,
            8.30369408e-26,
            1.24303033e-25,
            2.27726918e-25,
            1.06330422e-24,
            5.55017673e-24,
            1.92068553e-23,
            3.31361011e-23,
            1.13575247e-22,
            1.75386416e-22,
            6.52680518e-22,
            2.05106011e-21,
            6.37285545e-21,
            4.16125535e-20,
            1.12709507e-19,
            5.75853420e-19,
            1.73333796e-18,
            2.70099890e-18,
            7.53254646e-18,
            1.38139433e-17,
            3.60081965e-17,
            8.08419977e-17,
            1.86378584e-16,
            5.36224556e-16,
            8.89404640e-16,
            2.34045104e-15,
            4.72168880e-15,
            6.84378992e-15,
            2.26898430e-14,
            3.10087652e-14,
            2.78081199e-13,
            1.06479577e-12,
            2.81002203e-12,
            4.22067092e-12,
            9.27095863e-12,
            1.54519738e-11,
            4.53347819e-11,
            1.35564441e-10,
            2.35242087e-10,
            4.45253545e-10,
            9.78613696e-10,
            1.53140922e-09,
            2.81648560e-09,
            6.70890436e-09,
            1.49724785e-08,
            5.59553565e-08,
            1.39510811e-07,
            7.64761811e-07,
            1.40723957e-06,
            4.97638863e-06,
            2.12817943e-05,
            3.26471410e-05,
            1.02599591e-04,
            3.18774179e-04,
            5.67297630e-04,
            9.22732716e-04,
            1.17445643e-03,
            3.59279384e-03,
            3.61936491e-02,
            6.39493416e-02,
            1.29304829e-01,
            1.72272215e-01,
            3.46945901e-01,
            5.02826602e-01,
            6.24800042e-01,
            7.38412957e-01,
            7.59931663e-01,
            7.73374421e-01,
            7.91421897e-01,
            8.29325597e-01,
            8.57718637e-01,
            8.73286061e-01,
            8.77056835e-01,
            8.93173768e-01,
            9.05435646e-01,
            9.17217910e-01,
            9.19119179e-01,
            9.24810910e-01,
            9.26306908e-01,
            9.28621233e-01,
            9.33855835e-01,
            9.37263027e-01,
            9.41651642e-01,
        ]
    )
    x = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
            26.0,
            27.0,
            28.0,
            29.0,
            30.0,
            31.0,
            32.0,
            33.0,
            34.0,
            35.0,
            36.0,
            37.0,
            38.0,
            39.0,
            40.0,
            41.0,
            42.0,
            43.0,
            44.0,
            45.0,
            46.0,
            47.0,
            48.0,
            49.0,
            50.0,
            51.0,
            52.0,
            53.0,
            54.0,
            55.0,
            56.0,
            57.0,
            58.0,
            59.0,
            60.0,
            61.0,
            62.0,
            63.0,
            64.0,
            65.0,
            66.0,
            67.0,
            68.0,
            69.0,
            70.0,
            71.0,
            72.0,
            73.0,
            74.0,
            75.0,
            76.0,
            77.0,
            78.0,
            79.0,
            80.0,
            81.0,
            82.0,
            83.0,
            84.0,
            85.0,
            86.0,
            87.0,
            88.0,
            89.0,
            90.0,
            91.0,
            92.0,
            93.0,
            94.0,
            95.0,
            96.0,
            97.0,
            98.0,
        ]
    )
    kl = KneeLocator(
        x,
        y,
        curve="convex",
        direction="increasing",
        online=True,
        interp_method=interp_method,
    )
    assert kl.knee == 73


def test_valid_curve_direction():
    """Test that arguments to curve and direction are valid"""
    with pytest.raises(ValueError):
        kl = KneeLocator(range(3), [1, 3, 5], curve="bad curve")

    with pytest.raises(ValueError):
        kl = KneeLocator(range(3), [1, 3, 5], direction="bad direction")


def test_find_shape():
    """Test that find_shape can detect the right shape of curve line"""
    x, y = dg.concave_increasing()
    direction, curve = find_shape(x, y)
    assert direction == "increasing"
    assert curve == "concave"
    x, y = dg.concave_decreasing()
    direction, curve = find_shape(x, y)
    assert direction == "decreasing"
    assert curve == "concave"
    x, y = dg.convex_decreasing()
    direction, curve = find_shape(x, y)
    assert direction == "decreasing"
    assert curve == "convex"
    x, y = dg.convex_increasing()
    direction, curve = find_shape(x, y)
    assert direction == "increasing"
    assert curve == "convex"

def test_pmsm_torque():
    """Test that the global knee is correctly identified in PMSM example"""

    speeds = np.array(
        [
            0.0,
            3.881305,
            7.76261,
            11.643915,
            15.52522,
            19.406525,
            23.28783,
            27.169136,
            31.05044,
            34.931744,
            38.81305,
            42.694355,
            46.57566,
            50.456966,
            54.338272,
            58.219574,
            62.10088,
            65.982185,
            69.86349,
            73.7448,
            77.6261,
            81.50741,
            85.38871,
            89.27001,
            93.15132,
            97.03262,
            100.91393,
            104.795235,
            108.676544,
            112.557846,
            116.43915,
            120.32046,
            124.20176,
            128.08307,
            131.96437,
            135.84567,
            139.72697,
            143.60829,
            147.4896,
            151.3709,
            155.2522,
            159.1335,
            163.01482,
            166.89612,
            170.77742,
            174.65872,
            178.54002,
            182.42134,
            186.30264,
            190.18394,
            194.06525,
            197.94656,
            201.82787,
            205.70917,
            209.59047,
            213.47177,
            217.35309,
            221.23439,
            225.11569,
            228.997,
            232.8783,
            236.75961,
            240.64091,
            244.52222,
            248.40352,
            252.28482,
            256.16614,
            260.04742,
            263.92874,
            267.81006,
            271.69135,
            275.57266,
            279.45395,
            283.33527,
            287.21658,
            291.09787,
            294.9792,
            298.86047,
            302.7418,
            306.6231,
            310.5044,
            314.3857,
            318.267,
            322.14832,
            326.02963,
            329.91092,
            333.79224,
            337.67352,
            341.55484,
            345.43616,
            349.31744,
            353.19876,
            357.08005,
            360.96136,
            364.84268,
            368.72397,
            372.6053,
            376.48657,
            380.3679,
            384.2492,
            388.1305,
            392.0118,
            395.89313,
            399.7744,
            403.65573,
            407.53702,
            411.41833,
            415.29965,
            419.18094,
            423.06226,
            426.94354,
            430.82486,
            434.70618,
            438.58746,
            442.46878,
            446.35007,
            450.23138,
            454.1127,
            457.994,
            461.8753,
            465.7566,
            469.6379,
            473.51923,
            477.4005,
            481.28183,
            485.16312,
            489.04443,
            492.92575,
            496.80704,
            500.68835,
            504.56964,
            508.45096,
            512.3323,
            516.21356,
            520.09485,
            523.9762,
            527.8575,
            531.7388,
            535.6201,
            539.5014,
            543.3827,
            547.26404,
            551.1453,
            555.0266,
            558.9079,
            562.78925,
            566.67053,
            570.5518,
            574.43317,
            578.31445,
            582.19574,
            586.0771,
            589.9584,
            593.83966,
            597.72095,
            601.6023,
            605.4836,
            609.36487,
            613.2462,
            617.1275,
            621.0088,
            624.89014,
            628.7714,
            632.6527,
            636.534,
            640.41534,
            644.29663,
            648.1779,
            652.05927,
            655.94055,
            659.82184,
            663.7032,
            667.5845,
            671.46576,
            675.34705,
            679.2284,
            683.1097,
            686.99097,
            690.8723,
            694.7536,
            698.6349,
            702.51624,
            706.3975,
            710.2788,
            714.1601,
            718.04144,
            721.9227,
            725.804,
            729.68536,
            733.56665,
            737.44794,
            741.3293,
            745.2106,
            749.09186,
            752.97314,
            756.8545,
            760.7358,
            764.61707,
            768.4984,
            772.3797,
        ]
    )
    torques = np.array(
        [
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1307.5026,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1300.7279,
            1293.9532,
            1293.9532,
            1293.9532,
            1287.1786,
            1287.1786,
            1280.404,
            1280.404,
            1273.6294,
            1266.8547,
            1266.8547,
            1260.0802,
            1253.3055,
            1253.3055,
            1246.5309,
            1246.5309,
            1239.7562,
            1232.9817,
            1226.207,
            1226.207,
            1219.4324,
            1212.6578,
            1212.6578,
            1205.8832,
            1199.1085,
            1199.1085,
            1192.3339,
            1185.5593,
            1178.7847,
            1172.01,
            1172.01,
            1165.2354,
            1158.4608,
            1151.6862,
            1151.6862,
            1144.9115,
            1138.137,
            1131.3623,
            1131.3623,
            1124.5876,
            1117.813,
            1111.0385,
            1111.0385,
            1104.2638,
            1097.4891,
            1090.7146,
            1083.94,
            1083.94,
            1077.1653,
            1070.3906,
            1063.6161,
            1056.8414,
            1056.8414,
            1050.0668,
            1043.2921,
            1043.2921,
            1036.5176,
            1029.7429,
            1022.9683,
            1022.9683,
            1016.19366,
            1009.41907,
            1002.6444,
            995.8698,
            995.8698,
            989.09515,
            989.09515,
            982.32056,
            975.5459,
            975.5459,
            968.7713,
            961.9967,
            955.22205,
            955.22205,
            948.44745,
            941.6728,
            941.6728,
            934.8982,
            928.12354,
            928.12354,
            921.34894,
            921.34894,
            914.5743,
            914.5743,
            907.7997,
            901.025,
            894.2504,
            894.2504,
            894.2504,
        ]
    )
    kl_interp1d = KneeLocator(
        speeds,
        torques,
        curve="concave",
        direction="decreasing",
        interp_method="interp1d",
    )
    kl_make_splrep = KneeLocator(
        speeds,
        torques,
        curve="concave",
        direction="decreasing",
        interp_method="make_splrep",
        smoothing_factor=2.0,
    )

    # kneed values
    max_idx_internal_interp1d = int(np.argmax(kl_interp1d.y_difference))
    max_idx_interp1d = -(max_idx_internal_interp1d + 1)

    max_idx_internal_make_splrep = int(np.argmax(kl_make_splrep.y_difference))
    max_idx_make_splrep = -(max_idx_internal_make_splrep + 1)

    knee_speed_interp1d = float(kl_interp1d.x[max_idx_interp1d])
    knee_speed_make_splrep = float(kl_make_splrep.x[max_idx_make_splrep])

    # source values
    source_knee_idx_interp1d = int(np.argmin(np.abs(speeds - knee_speed_interp1d)))
    source_knee_idx_make_splrep = int(np.argmin(np.abs(speeds - knee_speed_make_splrep)))

    source_knee_speed_interp1d = speeds[source_knee_idx_interp1d]
    source_knee_speed_make_splrep = speeds[source_knee_idx_make_splrep]

    assert knee_speed_interp1d == pytest.approx(434.0, rel=0.015)
    assert knee_speed_make_splrep == pytest.approx(434.0, rel=0.015)
    assert source_knee_speed_interp1d == pytest.approx(434.0, rel=0.015)
    assert source_knee_speed_make_splrep == pytest.approx(434.0, rel=0.015)