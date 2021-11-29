import pytest

import judo


class TestNotebook:
    def test_remove_notebook_margin_not_crashes(self):
        judo.functions.notebook.remove_notebook_margin()
