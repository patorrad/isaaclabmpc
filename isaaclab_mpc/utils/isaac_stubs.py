"""Stub out Isaac Lab / Isaac Sim so example modules can be imported without a simulator.

Example planners (e.g. examples/*/planner.py) create an AppLauncher at import
time and their scene/robot modules import ``isaaclab.*`` at module level.  The
cost code itself is pure torch, so offline tools (cost viewers, tests) can
import those modules unchanged after calling ``install_isaac_stubs()``.

Every attribute of a stubbed module is a permissive class:
  - calling it records kwargs as attributes, so config trees stay readable
    (``CuboidCfg(size=(1, 2, 3)).size == (1, 2, 3)``);
  - class/instance attribute lookups return further stubs
    (``AssetBaseCfg.InitialStateCfg``, ``AppLauncher(args).app``);
  - it can be subclassed and used as a decorator.

``isaacsim`` is deliberately blocked (ImportError) so code that guards
optional debug drawing with ``try: import isaacsim...`` falls back cleanly.
"""

import importlib.abc
import importlib.machinery
import sys
import types

_STUB_ROOTS = ("isaaclab", "omni", "pxr", "carb")
_BLOCKED_ROOTS = ("isaacsim",)


class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _make_stub(name)


class _StubBase(metaclass=_StubMeta):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        # Decorator use (``@configclass class Foo``): behave as identity.
        self._wrapped = args[0] if len(args) == 1 and isinstance(args[0], type) else None

    def __call__(self, *args, **kwargs):
        if self._wrapped is not None:
            return self._wrapped(*args, **kwargs)
        return _make_stub("result")()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _make_stub(name)()

    def replace(self, **kwargs):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        new.__dict__.update(kwargs)
        return new


def _make_stub(name: str) -> type:
    return _StubMeta(name, (_StubBase,), {})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        stub = _make_stub(name)
        setattr(self, name, stub)
        return stub


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in _BLOCKED_ROOTS:
            raise ImportError(f"{fullname} is blocked by isaac_stubs")
        if root in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        mod = _StubModule(spec.name)
        mod.__path__ = []
        return mod

    def exec_module(self, module):
        pass


def install_isaac_stubs() -> None:
    """Install the stub finder (idempotent). Call before importing example modules."""
    if any(isinstance(f, _StubFinder) for f in sys.meta_path):
        return
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_ROOTS + _BLOCKED_ROOTS:
            del sys.modules[name]
    sys.meta_path.insert(0, _StubFinder())
