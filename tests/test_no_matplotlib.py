import sys
import pytest
import kneed.knee_locator as knee_locator_module
from kneed.data_generator import DataGenerator as dg
from kneed.knee_locator import KneeLocator


@pytest.mark.parametrize("method_name", ["plot_knee_normalized", "plot_knee"])
def test_plotting_methods_raise_when_matplotlib_is_missing(monkeypatch, method_name):
    """Test that plotting methods raise when matplotlib is unavailable."""
    monkeypatch.setattr(knee_locator_module, "_has_matplotlib", False, raising=False)
    monkeypatch.setattr(
        knee_locator_module,
        "_matplotlib_not_found_err",
        ModuleNotFoundError("matplotlib is required for plotting"),
        raising=False,
    )

    x, y = dg.figure2()
    kl = KneeLocator(x, y, S=1.0, curve="concave", interp_method="interp1d")

    with pytest.raises(ModuleNotFoundError):
        getattr(kl, method_name)()


# this is for running the test with `run and debug` in VSCode
# just add breakpoints, hit run and have fun. :)
if __name__ == "__main__":

    sys.exit(
        pytest.main(
            ["--log-cli-level=INFO", "--import-mode=importlib", "-vv", __file__]
        )
    )

    # Run only the specified test
    # sys.exit(
    #     pytest.main(
    #         [
    #             "--log-cli-level=INFO",
    #             "--import-mode",
    #             "importlib",
    #             "-vv",
    #             __file__,
    #             "-k",
    #             "test_logistic",
    #             "-o",
    #             "addopts=",
    #         ]
    #     )
    # )