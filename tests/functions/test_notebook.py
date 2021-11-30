import pytest

import judo


try:
    import IPython

    MISSING_IPYTHON = False
except ImportError:
    MISSING_IPYTHON = True


class TestNotebook:
    @pytest.mark.skipif(MISSING_IPYTHON, reason="IPython not installed")
    def test_remove_notebook_margin_not_crashes(self):
        judo.functions.notebook.remove_notebook_margin()

    def test_running_in_ipython_not_crashes(self):
        judo.functions.notebook.running_in_ipython()
