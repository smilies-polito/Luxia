"""
Root conftest for tests/python/.

The [tool.pytest.ini_options] pythonpath entry already adds this directory
to sys.path, so subdirectory test files can do:
    from utils.mesh_builders import make_clements_mesh_params
"""
