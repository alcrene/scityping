import sys
import re
import logging
from collections import defaultdict
from itertools import tee

logger = logging.getLogger(__name__)

class NotFound:
    pass

class UnsafeDeserializationError(RuntimeError):
    pass

def deref_module_name(val):
    """
    If `val` is a string, assume it is an import path to that module and
    return that module.
    Otherwise, return `val`.
    """
    if "." not in val:
        raise NotImplementedError(
            "I'm not sure in which situation one would need a str value "
            "that doesn't include a module; waiting for a use case.")
    else:
        # 'val' is a dotted name: first part indicates the package
        # where type is defined.
        # Problem: given "foo.bar.ObjName", we don't know whether to do
        # `from foo.bar import ObjName` or `from foo import bar.ObjName`.
        # So we try all combinations, giving precedence to more
        # specified packages (so `from foo.bar` would be tried first).
        # For each combination, we first check whether that package is
        # imported:  if not, it is not possible (without irresponsible
        # manipulation of imports) for that path to `ObjName` to be
        # defined.
        modname = val
        name = ""
        for k in range(val.count(".")):
            modname, name_prefix = val.rsplit(".", 1)
            name = name_prefix + ("." + name if name else "")
            mod = sys.modules.get(modname)
            if mod:  # Only attempt to import from already loaded modules
                try:
                    val = getattr(mod, name)
                except AttributeError:
                    pass  # Continue trying with a less specified import path
                else:
                    break
        else:
            # If we exit the loop without finding anything, it means the
            # required module is not available.
            raise ModuleNotFoundError(f"The value {val} does not match "
                                      "any imported module.")

    return val

class ModuleList(list):
    def __iter__(self):
        """
        Return elements from the list. Strings are replaced by importing the
        corresponding module.
        """
        baseiter = super().__iter__()
        for i, el in enumerate(baseiter):
            if isinstance(el, str):
                el = deref_module_name(el)
                # Update list so module is not imported again
                self[i] = el
            yield el

class LazyDict(dict):
    """A dictionary allowing to specify objects that haven’t been loaded yet.

    A string value is interpreted as the import path to an object. This is
    useful for specifying types without requiring them to be imported, or which
    may not always be available.

    .. Caution:: Should not be used to store `str` values which are not imports.
    """
    def __getitem__(self, key):
        val = super().__getitem__(key)
        if isinstance(val, str):
            newval = deref_module_name(val)
            # Store value so it doesn’t need to be imported again
            self[key] = newval
            val = newval
        return val

    def get(self, key, default=None):
        val = super().get(key, default)
        if val != default and isinstance(val, str):  # I think it’s less surprising if the default is always unmodified ?
            try:
                newval = deref_module_name(val)
            except ModuleNotFoundError:
                return default  # EARLY EXIT
            # Store value so it doesn’t need to be imported again
            self[key] = newval
            val = newval
        return val


def get_type_key(obj: type):
    try:
        serialization_key = obj.__qualname__
        module = obj.__module__
    except AttributeError:
        raise TypeError("`get_type_key` accepts only instances of `type`.")
    # Allow types to override their serialization key; this is used by generics like PureFunction
    serialization_key = getattr(obj, "__serialization_key__", serialization_key)
    return f"{module}.{serialization_key}"

class TypeRegistry(dict):
    """A registry mapping serializable type keys to the types themselves.

    A `TypeRegistry` attached to serializable type A is used to store the list
    of serialiable types which are compatible with A (loosely speaking, values
    of the registry are subtypes of A).
    To allow fuzzy matching, we associate each subtype with a string key which
    is all lowercased and can be split into tokens separated by a period ".".
    Keys are computed deterministically from a type, and there can be multiple
    keys pointing to the same serializable type; this allows a single
    serializer to be used for multiple types.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._key_tokens = {}
        for k, v in dict(*args, **kwargs).items():
            self[k] = v  # Use __setitem__ which has the key standardization

    def __contains__(self, key):
        """Implement case-insensitive key matching."""
        if isinstance(key, type):
            key = get_type_key(key)
        elif not isinstance(key, str):
            raise TypeError("`TypeRegistry` accepts only types or strings "
                            f"as keys; received: {key}")
        return super().__contains__(key.lower())

    def __setitem__(self, key, value):
        if isinstance(key, type):
            key = get_type_key(key)
        elif not isinstance(key, str):
            raise TypeError("`TypeRegistry` accepts only types or strings "
                            f"as keys; received: {key}")
        key = key.lower()
        tokens = frozenset(tok for tok in key.split("."))
        if tokens in self._key_tokens:
            if self._key_tokens[tokens] == key and self.get(key.lower()) is value:
                # Some solutions for special cases may be  easier if we allow
                # setting registry values more than once. As long as the value
                # is the same, this is fine.
                return  # EARLY EXIT
            else:
                logger.error("The Serializable TypeRegistry already contained an "
                             f"entry for key {self._key_tokens[tokens]}, which "
                             "has been replaced.")
                super().__delitem__(self._key_tokens[tokens])
        super().__setitem__(key, value)
        self._key_tokens[tokens] = key

    def __delitem__(self, key):
        if isinstance(key, type):
            key = get_type_key(key)
        elif not isinstance(key, str):
            raise TypeError("`TypeRegistry` accepts only types or strings "
                            f"as keys; received: {key}")
        key = key.lower()
        tokens = frozenset(tok for tok in key.split("."))
        super().__delitem__(key)
        del self._key_tokens[tokens]

    def __getitem__(self, key):
        if isinstance(key, type):
            key = get_type_key(key)
        elif not isinstance(key, str):
            raise TypeError("`TypeRegistry` accepts only types or strings "
                            f"as keys; received: {key}")
        key = key.lower()

        # Fast initial search: use subset matching to find a) the largest number
        # of tokens which match some of the keys, and b) the corresponding keys
        ordered_tokens = tuple(tok for tok in  key.split("."))
        tokens = set(ordered_tokens)
        matches = defaultdict(lambda: [])
        for regtokens, regkey in self._key_tokens.items():
            if regtokens & tokens:
                matches[len(regtokens & tokens)].append((regtokens, regkey))

        # Slower filtering based on matching RTL
        if not matches:
            raise KeyError(f"Registry contains no entry matching {key}")
        else:
            matches = matches[max(matches)]  # Look only at keys which match the most tokens
            pattern = r"(?:[a-zA-Z0-9_\-.[]()]+\.|)"  # Pattern allows for missing modules between tokens
            rtl_matches = [regkey for (regtokens, regkey) in matches if
                           re.search(pattern  # NB: We prepend 'pattern' but don't postpend it (the actual type name should match)
                                     + pattern.join(tok.replace("[",r"\[").replace("(",r"\)") + "."
                                                    for tok in ordered_tokens[:-1]
                                                    if tok in (regtokens & tokens))
                                     + ordered_tokens[-1].replace("[",r"\[").replace("(",r"\)")
                                     + "$",  # The rightmost token must always be included 
                                     regkey.lower())]
            if len(rtl_matches) == 0:
                raise KeyError(f"No entry for key '{key}'. Closest matches: "
                               f"{[regkey for regtokens, regkey in matches]}.")
            elif len(rtl_matches) > 1:
                # Problem: a longer path can shadow a shorter one
                # Solution: if there is an exact match, pick that one
                exact_matches = [m for m in rtl_matches if m == key]
                if len(exact_matches) == 1:
                    rtl_matches = exact_matches
                else:
                    # Don’t use KeyError, because then there may be no way do distinguish this
                    # case from a missing case (relied upon by e.g. `get`)
                    raise ValueError("Unable to unambiguously match a registry "
                                     f"entry to '{key}'. Registry contains: {rtl_matches}.")

            return super().__getitem__(rtl_matches[0])

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

