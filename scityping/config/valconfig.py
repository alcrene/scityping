"""
Utilities for creating project config objects

For a package called `MyPackage`, the following files should be defined:

    MyPackage/MyPackage/config.py
    MyPackage/.project-defaults.cfg

Within MyPackage/config.py, one then does something like

    from pathlib import Path
    from pydantic import BaseModel
    from mackelab_toolbox.config import ValConfig

    class Config(ValConfig):
        class PATH(BaseModel):
            <path param name 1>: <type 1>
            <path param name 2>: <type 2>
            ...
        class RUN(BaseModel):
            <run param name 1>: <type 1>
            <run param name 2>: <type 2>
            ...

    root = Path(__file__).parent.parent
    config = Config(
        path_user_config   =root/"project.cfg",
        path_default_config=root/".project-default.cfg",
        package_name       ="MyPackage")

(c) Alexandre René 2022-2023
https://github.com/mackelab/mackelab-toolbox/tree/master/mackelab_toolbox/config
"""
import os
import logging
from pathlib import Path
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Optional, Union, ClassVar#, _UnionGenericAlias  >= 3.9
from configparser import ConfigParser, ExtendedInterpolation
from pydantic import BaseModel, validator
from pydantic.main import ModelMetaclass
import textwrap

logger = logging.getLogger(__name__)

class ValConfigMeta(ModelMetaclass):
    """
    Some class magic with nested types:
    1. If a nested type is also used to declare a value or annotation, it is
       left untouched.
    2. If a nested type is declared but not used, do the following:
       1. Convert it to a subclass of `BaseModel` if it isn't already one.
          This allows concise definition of nested configuration blocks.
       2. Declare an annotation with this type and the same name.
          ('Type' is appended to the attribute declaring the original type,
          to prevent name conflicts.)
          Exception: If "<typename>Type" is already used in the class, and thus
          would cause conflicts, no annotation is added.
    """
    def __new__(metacls, cls, bases, namespace):
        # Use ValConfig annotations as default. That way users don't need to remember to type `package_name: str = "MyProject"`
        # However, in order not to lose the default if the user *didn't* assign to that attribute,
        # we only use annotation defaults for values which are also in `namespace`.
        default_annotations = {} if cls  == "ValConfig" \
                                 else ValConfig.__annotations__
        annotations = {**{nm: ann for nm, ann in default_annotations.items()
                          if nm in namespace},
                       **namespace.get("__annotations__", {})}
        if annotations:
            # Unfortunately a simple `Union[annotations.values()].__args__` does not work here
            def deref_annotations(ann):
                if isinstance(ann, Iterable):
                    for a in ann:
                        yield deref_annotations(a)
                elif hasattr(ann, "__args__"):
                    for a in ann.__args__:
                        yield deref_annotations(a)
                else:
                    yield ann
            annotation_types = set(deref_annotations(T) for T in annotations.values())
        else:
            annotation_types = set()
        attribute_types = set(type(v) for v in namespace.values())
        nested_classes = {nm: val for nm, val in namespace.items()
                          if isinstance(val, type) and nm not in {"Config", "__config__"}}
        new_namespace = {nm: val for nm, val in namespace.items()
                         if nm not in nested_classes}
        new_nested_classes = {}
        for nm, T in nested_classes.items():
            # If a declared type was used, don't touch it or its name, and don't create an associated attribute
            if T in annotation_types | attribute_types:
                new_nested_classes[nm] = T
                continue
            # Otherwise, append `Type` to the name, to free the name itself for an annotation
            # NB: This only renames the nested attribute, not the type itself
            new_nm = nm + "Type"
            if new_nm in annotations.keys() | new_namespace.keys():
                new_nm = nm  # Conflict -> no rename
            # If it isn't already a subclass of BaseModel, make it one
            if T.__bases__ == (object,):
                copied_attrs = {nm: attr for nm, attr in T.__dict__.items()
                                if nm not in {'__dict__', '__weakref__', '__qualname__', '__name__'}}
                newT = ValConfigMeta(nm, (ValConfig,), copied_attrs)
                # newT = type(nm, (T,BaseModel), {})  
            else:
                if not issubclass(T, ValConfig):
                    logger.warning(f"For the nested Config class '{T.__qualname__}' "
                                   "to be automatically converted to a subclass of `BaseModel`, "
                                   "it must not inherit from any other class.")
                newT = T
            new_nested_classes[new_nm] = newT
            # Add a matching annotation
            if new_nm != nm:  # Ensure we aren't overwriting the type
                annotations[nm] = newT

        return super().__new__(metacls, cls, bases,
                               {**new_namespace, **new_nested_classes,
                                '__annotations__': annotations,
                                '__valconfig_initialized__': False})


# Singleton pattern
_config_instances = {}

class ValConfig(BaseModel, metaclass=ValConfigMeta):
    """
    Augments Python's ConfigParser with a dataclass interface and automatic validation.
    Pydantic is used for validation.

    The following package structure is assumed:

        code_directory
        ├── .gitignore
        ├── setup.py
        ├── project.cfg
        └── MyPkcg
            ├── [code files]
            └── config
                ├── __init__.py
                ├── defaults.cfg
                └── [other config files]

    `ValConfig` should be imported and instantiated from within
    ``MyPckg.config.__init__.py``::

       from pathlib import Path
       from mackelab_toolbox.config import ValConfig

       class Config(ValConfig):
           arg1: <type>
           arg2: <type>
           ...

       config = Config(Path(__file__).parent/".project-defaults.cfg")

    `project.cfg` should be excluded by `.gitignore`. This is where users can
    modify values for their local setup. If it does not exist, a template one
    is created from the contents of `.project-defaults.cfg`, along with
    instructions.

    There are some differences and magic behaviours compared to a plain
    BaseModel, which help to reduce boilerplate when defining configuration options:
    - Defaults are validated (`validate_all = True`).
    - Values of ``"<default>"`` are replaced by the hard coded default in the Config
      definition. (These defaults may be `None`.)
    - Nested plain classes are automatically converted to inherit ValConfig,
      and a new attribute of that class type is created. Specifically, if we
      have the following:

          class Config(ValConfig):
              class paths:
                  projectdir: Path

      then this is automatically converted to

          class Config(ValConfig):
              class pathsType:
                  projectdir: Path

              path : pathsType
    - The user configuration file is found by searching upwards from the current
      directory for a file matching the value of `Config.user_config_filename`
      (default: "project.cfg")
      + If multiple config files are found, **the most global one is used**.
        The idea of a global config files is to help provide consistency across
        a document. If a sub project is compiled as part of a larger one, we want
        to use the larger project's config, which may set things like font
        sizes and color schemes, for all figures/report pages.
      + If it is important that particular pages use particular options, the
        global config file is not the best place to set that. Rather, set
        those options in the file itself, or a sibling text file.
    # - If a user configuration file is not found, and the argument
    #   `ensure_user_config_exists` is `True`, then a new blank configuration file
    #   is created at the location we would have expected to find one: inside
    #   the nearest parent directory which is a version control repository.
    # - The `rootdir` value is the path of the directory containing the user config.
    # - All arguments of type `Path` are made absolute by prepending `rootdir`.
    #   Unless they already are absolute, or the class variable
    #   `make_paths_absolute` is set to `False`.
    #   !! NOT CURRENTLY TRUE !!: For reasons I don’t yet understandand, the
    #   mechanism to do this adds the correct validator, but it isn’t executed.
    #   For the moment, please import the `prepend_rootdir` function and apply it
    #   (wrapping with ``validator(field)(prepend_rootdir)`` to all relevant fields.


    Config class options
    --------------------

    __create_template_config: Set to `True` if you want a template config
        file to be created in a standard location when no user config file is
        found. Default is `False`. Typically, this is set to `False` for utility
        packages, and `True` for project packages.
    __interpolation__: Passed as argument to ConfigParser.
        Default is ExtendedInterpolation().
        (Note that, as with ConfigParser, an *instance* must be passed.)
    __empty_lines_in_values__: Passed as argument to ConfigParser.
        Default is True: this prevents multiline values with empty lines, but
        makes it much easier to indent without accidentally concatenating values.

    Inside config/__init__.py, one would have:

        config = Config(Path(__file__)/".project-defaults.cfg" 
    """
    # rootdir: Path

    # package_name: ClassVar[str]
    # default_config_file: ClassVar[Union[str,Path]]
    # ensure_user_config_exists: ClassVar[bool]=False  # Set to True to add a template config file when none is found

    #path_default_config: Path="config/.project-defaults.cfg"  # Rel path => prepended with rootdir
    #path_user_config: Path="../project.cfg"  # Rel path => prepended with rootdir
    # user_config_filename: ClassVar[str]="project.cfg"  # Searched for, starting from CWD

    ## Config class options ##
    # Class options use dunder names to avoid conflicts
    # __make_paths_absolute__  : ClassVar[bool]=True
    __default_config_path__   : ClassVar[Optional[Path]]=None
    __local_config_filename__ : ClassVar[Optional[str]]=None
    __create_template_config__: ClassVar[bool] = False  # If True, a template config file is created when no local file is found. Required a default config set with __default_config_path__
    __interpolation__         : ClassVar = ExtendedInterpolation()
    __emtpy_lines_in_values__ : ClassVar = False
    __top_message_default__: ClassVar = """
        # This configuration file for '{package_name}' should be excluded from
        # git, so can be used to configure machine-specific variables.
        # This can be used for example to set output paths for figures, or to
        # set flags (e.g. using GPU or not).
        # Default values are listed below; uncomment and edit as needed.
        #
        # Adding a new config field is done by modifying the config module `{config_module_name}`.
        
        """
    # Internal vars
    __valconfig_current_root__: ClassVar[Optional[Path]]=None  # Set temporarily when validating config files, in order to resolve relative paths
    __valconfig_deferred_init_kwargs__: ClassVar[list]=[]

    ## Pydantic model configuration ##
    # `validation_assignment` must be True, other options can be changed.
    class Config:
        validate_all = True  # To allow specifying defaults with as little boilerplate as possible
                             # E.g. without this, we would need to write `mypath: Path=Path("the/path")`
        validate_assignment = True  # E.g. if a field converts str to Path, allow updating values with strings

    ## Singleton pattern ##
    def __new__(cls, *a, **kw):
        if cls not in _config_instances:
            _config_instances[cls] = super().__new__(cls)  # __init__ will add this to __instances
        return _config_instances[cls]
    def __copy__(x):  # Singleton => no copies
        return x
    def __deepcopy__(x, memo=None):
        return x

    ## Interface ##
    def __dir__(self):
        return list(self.__fields__)

    ## Initialization ##

    # TODO: Config file as argument instead of cwd ?
    def __init__(self,
                 # default_config_file: Union[None,str,Path]=None,
                 cwd: Union[None,str,Path]=None,
                 # ensure_user_config_exists: bool=False,
                 # rootdir: Union[str,Path],
                 # path_default_config=None, path_user_config=None,
                 # *,
                 # config_module_name: Optional[str]=None,
                 **kwargs
                 ):
        """
        Instantiate a `Config` instance, reading from both the default and
        user config files.
        If the user-editable config file does not exist yet, an empty one
        with instructions is created, at the root directory of the version-
        controlled repository. If there are multiple nested repositories, the
        outer one is used (logic: one might have a code repo outside a project
        repo; this should go in the project repo). If no repository is found,
        no template config file is created.


        See also `ValConfig.ensure_user_config_exists`.
        
        Parameters
        ----------
        # default_config_file: Path to the config file used for defaults.
        #     SPECIAL CASE: If this value is None, then `ValConfig`
        #     behaves like `ValConfigBase`: no config file is parsed or
        #     created. This is intended for including as a component of a larger
        #     ValConfig.
        #     The assumption then is that all fields are passed by keywords.
        #     NOT TRUE: Currently we do the special case if kwargs is non-empty.
        cwd: "Current working directory". The search for a config file starts
            from this directory then walks through the parents. 
            The value of `None` indicates to use the current working directory;
            especially if running on a local machine, this is generally what
            you want, and is usually the most portable.
        *
        # config_module_name: The value of __name__ when called within the
        #     project’s configuration module.
        #     Used for autogenerated instructions in the template user config file.

        # **kwargs:
        #     Additional keyword arguments are passed to ConfigParser.

        Todo
        ----
        When multiple config files are present in the hierarchy, combine them.
        Relative paths in lower config files should still work.
        """
        if self.__valconfig_initialized__:
            # Already initialized; if there are new kwargs, validate them.
            self.validate_dict(kwargs)
            return

        elif kwargs:
            # We have NOT yet initialized, but are passing keyword arguments:
            # this may happen because of __new__ returning an existing instance,
            # in which case this __init__ gets executed twice
            # We flush this list after we initialize
            self.__valconfig_deferred_init_kwargs__.append(kwargs)

        else:
            # Normal path: Initialize with the config files

            ## Read the default config file ##
            if self.__default_config_path__:
                # This will use super().__init__ because __valconfig_initialized__ is still False
                self.validate_cfg_file(self.__default_config_path__)
            else:
                # If there is no config file, then all defaults must be defined in Config definition
                super().__init__()

            # Mark as initialized, so we don't take this branch twice
            # Also, this ensures that further config files use setattr() instead of __init__ to set fields
            type(self).__valconfig_initialized__ = True  # Singleton => Assign to class

            ## Read the local (user-specific) config file(s) ##
            # Any matching file in the hierarchy will be read; files deeper in the file hierarchy are read last, so have precedence
            if self.__local_config_filename__:
                cfg_fname = self.__local_config_filename__

                ## Search for a file with name matching `cfg_fname` in the current directory and its parents ##
                # If no project config is found, create one in the location documented above
                if cwd is None:
                    cwd = Path(os.getcwd())
                default_location_for_conf_filename = None  # Will be set the first time we find a .git folder
                cfg_paths = []
               
                for wd in [cwd, *cwd.parents]:
                    wdfiles = set(os.listdir(wd))
                    if cfg_fname in wdfiles:
                        rootdir = wd
                        cfg_paths.append(cwd/cfg_fname)
                        # break
                    if ({".git", ".hg", ".svn"} & wdfiles
                          and not default_location_for_conf_filename):
                        default_location_for_conf_filename = wd/cfg_fname

                if rootdir:
                    for cfg_path in reversed(cfg_paths):
                        self.validate_cfg_file(cfg_path)
                elif default_location_for_conf_filename is not None:
                    if self.__create_template_config__:
                        # We didn’t find a project config file, but we did find that
                        # we are inside a VC repo => place config file at root of repo
                        # `ensure_user_config_exists` creates a config file from the
                        # defaults file, listing all default values (behind comments),
                        # and adds basic instructions and the default option values
                        assert self.__default_config_path__, f"Cannot create a template file at {default_location_for_conf_filename} if `{type(self).__qualname__}.__default_config_path__` is not set."
                        self.add_user_config_if_missing(
                            self.__default_config_path__,
                            default_location_for_conf_filename)
                elif self.__create_template_config__:
                    logger.error(f"The provided current working directory ('{cwd}') "
                                 "is not part of a version controlled repository.")

            ## Apply any deferred initialization kwargs ##
            for kw in self.__valconfig_deferred_init_kwargs__:
                self.validate_dict(kw)
            self.__valconfig_deferred_init_kwargs__.clear()

    def validate_cfg_file(self, cfg_path: Path,
                          empty_lines_in_values=False):
        cfp = ConfigParser(interpolation=self.__interpolation__,
                           empty_lines_in_values=empty_lines_in_values)
        cfp.read(cfg_path)

        # Convert cfp to a dict; this loses the 'defaults' functionality, but makes
        # it much easier to support validation and nested levels
        cfdict = {section: dict(values) for section, values in cfp.items()}

        # Validate as a normal dictionary
        # We set the current root so that relative paths are resolved based on
        # the location of their config file
        type(self).__valconfig_current_root__ = cfg_path.parent.absolute()
        self.validate_dict(cfdict)
        type(self).__valconfig_current_root__ = None


    def validate_dict(self, cfdict):
        """
        Note: This function relies on `validate_assignment = True`
        """
        # Convert any dotted sections into dict hierarchies (and merge where appropriate)
        cfdict = _unflatten_dict(cfdict)

        # Use "one-level-up" as a default value
        # So if there is no "pckg.colors" section, but there is a "colors" section,
        # use "colors" for "pckg.colors", if pckg expects that section.
        for fieldnm, field in self.__fields__.items():
            # Having Union[ValConfig, other type(s)] could break the logic of the next test; since we have no use case, just detect, warn and exit
            # if isinstance(field.type_, _UnionGenericAlias):  ≥3.9
            if str(field.type_).startswith("typing.Union"):
                if any((isinstance(T, type) and issubclass(T, ValConfig))
                       for T in field.type_.__args__):
                    raise TypeError("Using a subclass of `ValConfig` inside a Union is not supported.")
            elif isinstance(field.type_, type) and issubclass(field.type_, ValConfig):
                if fieldnm not in cfdict:
                    cfdict[fieldnm] = {}
                else:
                    assert isinstance(cfdict[fieldnm], dict), f"Configuration field '{fieldnm}' should be a dictionary."  # Must be mutable
                subcfdict = cfdict[fieldnm]
                for subfieldnm, subfield in field.type_.__fields__.items():
                    if subfieldnm not in subcfdict and subfieldnm in cfdict:
                        # A field is missing, and a plausible default is available
                        # QUESTION: Should we make a (deep) copy, in case two validating configs make conflicting changes ?
                        subcfdict[subfieldnm] = cfdict[subfieldnm]

                # If we didn’t add anything, remove the blank dictionary
                if len(cfdict[fieldnm]) == 0:
                    del cfdict[fieldnm]

        if self.__valconfig_initialized__:
            # Config has already been initialized => validate fields by setting them
            # This relies on `validate_assignment = True`
            recursively_validate(self, cfdict)
        else:
            # We are initializing => Use __init__ to validate values
            super().__init__(**cfdict)

    ## Validators ##

    @validator("*", pre=True)
    def prepend_rootdir(cls, val, values):
        """Prepend any relative path with the current root directory."""
        if isinstance(val, Path) and not val.is_absolute():
            root = cls.__valconfig_current_root__
            if root:
                val = root/val
        return val

    @validator("*", pre=True)
    def reset_default(cls, val, field):
        """Allow to use defaults in the BaseModel definition.

        Config models can define defaults. To allow config files to specify
        that we want to use the hardcoded default, we interpret the string value
        ``"<default>"`` to mean to use the hardcoded default.
        """
        if val == "<default>":
            val = field.default  # NB: If no default is set, this returns `None`
        return val

    ## User config template ##

    def add_user_config_if_missing(
        self,
        path_default_config: Union[str,Path],
        path_user_config: Union[str,Path],
        ):
        """
        If the user-editable config file does not exist, create it.

        Basic instructions are added as a comment to the top of the file.
        Their content is determined by the class variable `__top_message_default__`.
        Two variables are available for substitution into this message:
        `package_name` and `config_module_name`.

        Parameters
        ----------
        path_default_config: Path to the config file providing defaults.
            *Should* be version-controlled
        path_user_config: Path to the config file a user would modify.
            Should *not* be version-controlled
        """
        # Determine the dynamic fields for the info message added to the top
        # of the template config
        package_name, _ = self.__module__.split(".", 1)
            # If the ValConfig subclass is defined in ``mypkg.config.__init__.py``, this will return ``mypkg``.
        config_module_name = self.__file__

        top_message = self.__top_message_default__
        # Remove any initial newlines from `top_message`
        for i, c in enumerate(top_message):
            if c != "\n":
                top_message = top_message[i:]
                break
        # Finish formatting top message
        top_message = textwrap.dedent(
            top_message.format(package_name=package_name,
                               config_module_name=config_module_name))

        if not Path(path_user_config).exists():
            # The user config file does not yet exist – create it, filling with
            # commented-out values from the defaults
            with open(path_default_config, 'r') as fin:
                with open(path_user_config, 'x') as fout:
                    fout.write(textwrap.dedent(top_message))
                    stashed_lines = []  # Used to delay the printing of instructions until after comments
                    skip = True         # Used to skip copying the top message from the defaults file
                    for line in fin:
                        line = line.strip()
                        if skip:
                            if not line or line[0] == "#":
                                continue
                            else:
                                # We've found the first non-comment, non-whitespace line: stop skipping
                                skip = False
                        if not line:
                            fout.write(line+"\n")
                        elif line[0] == "[":
                            fout.write(line+"\n")
                            stashed_lines.append("# # Defaults:\n")
                        elif line[0] == "#":
                            fout.write("# "+line+"\n")
                        else:
                            for sline in stashed_lines:
                                fout.write(sline)
                            stashed_lines.clear()
                            fout.write("# "+line+"\n")
            logger.warning(f"A default project config file was created at {path_user_config}.")


## Convenience validators ##

import os

def ensure_dir_exists(cls, dirpath):
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    return dirpath

## Utilities ##

def _unflatten_dict(d: Mapping) -> defaultdict:
    """Return a new dict where dotted keys become nested dictionaries.

    For example, ``{"a.b.c.d": 3}`` becomes
    ``{"a": {"b": {"c": {"d": 3}}}}``

    Keys are sorted before unflattening, so the order of items in `d` does not matter.
    Precedence is given to later, more specific keys. For example, the following
    dictionary::

       {"a": {"b1": 1, "b2": {"c": 3}},
        "a.b1": 10,
        "a.b2": {"c": 30},
        "a.b2.c": 300
        }

    would become::

       {"a": {"b1": 10, "b2": {"c": 300}}}

    Note how in this example we used::

        "a.b2": {"c": 30},

    All intermediate levels must be dictionaries, even if those values are
    ultimately not used. Otherwise an `AssertionError` is raised.

    """
    def new_obj(): return defaultdict(new_obj)
    obj = new_obj()

    # Logic inspired by https://github.com/mewwts/addict/issues/117#issuecomment-756247606
    for k in sorted(d.keys()):  # Use sorted keys for better reproducibility
        v = d[k]
        subks = k.split('.')
        last_k = subks.pop()
        for i, _k in enumerate(subks):
            obj = obj[_k]
            assert isinstance(obj, Mapping), \
                f"Configuration field '{'.'.join(subks[:i+1])}' should be a dictionary."
        # NB: Don't unflatten value dictionaries, otherwise we can't have configs
        #     like those in matplotlib: {'figure.size': 6}
        obj[last_k] = v

    return obj

def recursively_validate(model: Union[BaseModel,ValConfig],
                         newvals: dict):
    for key, val in newvals.items():
        # If `val` is a Mapping, we want to assign it recursively
        # if the field is a nested Config.
        if isinstance(val, Mapping):
            cur_val = getattr(model, key)
            # Two ways to identify nested Config: has `validate_dict` method
            if hasattr(cur_val, "validate_dict"):
                curv_val.validate_dict(val)
            # or is an instance of BaseModel
            elif isinstance(cur_val, BaseModel):
                recursively_validate_basemodel(cur_val, val)
            # Note that we don’t recursively validate plain dicts: it’s not unlikely one would want to replace them with the new value
            else:
                setattr(model, key, val)
        else:
            setattr(model, key, val)  
