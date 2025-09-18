# pyfx_utils/ipyext.py
import shlex

def load_ipython_extension(ipython):
    from .colab_boot import boot

    def _pyfx_boot(line=""):
        # Optional args: code_dir=..., ensure=pkg1,pkg2
        kwargs = {}
        for tok in shlex.split(line):
            if "=" in tok:
                k, v = tok.split("=", 1)
                if k == "code_dir":
                    kwargs["code_dir"] = v
                elif k == "ensure":
                    kwargs["ensure"] = tuple([p.strip() for p in v.split(",") if p.strip()])

        ns = boot(**kwargs)
        ipython.user_global_ns.update(ns)

        # nice-to-have: auto-reload edited modules
        try:
            ipython.run_line_magic("load_ext", "autoreload")
            ipython.run_line_magic("autoreload", "2")
        except Exception:
            pass

        print("✅ pyfx booted:", ", ".join(sorted(ns.keys())))

    def _pyfx_reload(line=""):
        import sys, importlib
        for m in list(sys.modules):
            if m.startswith("pyfx_utils"):
                del sys.modules[m]
        importlib.invalidate_caches()
        print("♻️ Cleared pyfx_utils modules & invalidated caches.")

    ipython.register_magic_function(_pyfx_boot,   magic_kind="line", magic_name="pyfx_boot")
    ipython.register_magic_function(_pyfx_reload, magic_kind="line", magic_name="pyfx_reload")

def unload_ipython_extension(ipython):
    # optional: nothing needed
    pass
