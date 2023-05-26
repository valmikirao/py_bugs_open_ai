from py_bugs_open_ai import constants
import setup_constants


def test_setup_constants():
    """
    There are a bunch of constants that I want to be usable by both setup.py and the rest of the app.  But python
    installation makes this tricky.  So I just copy the constants in two location and use this test to make sure
    they're values are the same
    """
    constant_names = list(
        c for c in dir(setup_constants) if not c.startswith('__') and c.upper() == c
    )

    for constant_name in constant_names:
        assert hasattr(constants, constant_name)
        assert getattr(constants, constant_name) == getattr(setup_constants, constant_name)
