import importlib
from typing import Text, Optional, Any


def class_from_module_path(
    module_path: Text, lookup_path: Optional[Text] = None
) -> Any:
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects."""
    # load the module, will raise ImportError if module cannot be loaded
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition(".")
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        return getattr(m, class_name)
    else:
        module = globals().get(module_path, locals().get(module_path))
        if module is not None:
            return module

        if lookup_path:
            # last resort: try to import the class from the lookup path
            m = importlib.import_module(lookup_path)
            return getattr(m, module_path)
        else:
            raise ImportError(f"Cannot retrieve class from path {module_path}.")
