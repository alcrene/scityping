"""
Utilities for creating project config objects

For a package called `MyPackage`, the following files should be defined:

    MyPackage/MyPackage/config.py
    MyPackage/.project-defaults.cfg

Within MyPackage/config.py, one then does something like

    from pathlib import Path
    from pydantic import BaseModel
    from mackelab_toolbox.config import ValidatingConfig

    class Config(ValidatingConfig):
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
from pathlib import Path
from typing import Optional, Union, ClassVar#, _UnionGenericAlias  >= 3.9
import logging
from configparser import ConfigParser, ExtendedInterpolation
from pydantic import BaseModel, validator
from pydantic.main import ModelMetaclass
from pydantic.utils import lenient_issubclass
import textwrap

logger = logging.getLogger(__name__)

class ValidatingConfigMeta(ModelMetaclass):
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
        # Use ValidatingConfig annotations as default. That way users don't need to remember to type `package_name: str = "MyProject"`
        # However, in order not to lose the default if the user *didn't* assign to that attribute,
        # we only use annotation defaults for values which are also in `namespace`.
        default_annotations = {} if cls in {"ValidatingConfig", "ValidatingConfigBase"} \
                                 else ValidatingConfig.__annotations__
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
                newT = ValidatingConfigMeta(nm, (ValidatingConfigBase,), copied_attrs)  # TODO?: Use Singleton here? (without causing metaclass conflict…)
                # newT = type(nm, (T,BaseModel), {})  
            else:
                if not issubclass(T, ValidatingConfigBase):
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
                                '__annotations__': annotations})


# Singleton pattern
_config_instances = {}

class ValidatingConfigBase(BaseModel, metaclass=ValidatingConfigMeta):
    """
    Same as ValidatingConfig, without the parsing of the config file.
    Mostly used for nested entries.
    """
    # Used for passing arguments to validators -- use names that won’t cause conflicts
    make_paths_absolute: ClassVar[bool]=True

    rootdir: Optional[Path]

    class Config:
        validate_all = True  # To allow specifying defaults with as little boilerplate as possible
                             # E.g. without this, we would need to write `mypath: Path=Path("the/path")`
        validate_assignment = True  # E.g. if a field converts str to Path, allow updating values with strings

    # Singleton pattern
    def __new__(cls, *a, **kw):
        if cls not in _config_instances:
            _config_instances[cls] = super().__new__(cls)  # __init__ will add this to __instances
        return _config_instances[cls]

    def __init__(self, *, rootdir, **cfdict):
        """

        Parameters
        ----------
            rootdir: By default, all relative paths are prepended with `rootdir`.
                This is the location 
        """
        # # Singleton pattern
        # if type(self) in ValidatingConfigBase.__instances:
        #     # Already initialized; nothing to do
        #     return

        # Convert dotted sections to nested dicts
        # We do one level here; recursion takes care of nested levels 
        tomove = []
        for section in cfdict:
            if "." in section:
                section_, subsection = section.split(".", 1)
                tomove.append((section_, subsection))
        for section_, subsection in tomove:
            if section_ not in cfdict:
                cfdict[section_] = {}
            else:
                assert isinstance(cfdict[section_], dict), f"Configuration field '{fieldnm}' should be a dictionary."  # Must be mutable
            if subsection not in cfdict[section_]:
                cfdict[section_][subsection] = cfdict[f"{section_}.{subsection}"]
                del cfdict[f"{section_}.{subsection}"]
        # Use "one-level-up" as a default value
        # So if there is no "pckg.colors" section, but there is a "colors" section,
        # use "colors" for "pckg.colors", if pckg expects that section.
        for fieldnm, field in self.__fields__.items():
            # Having Union[ValidatingConfig, other type(s)] could break the logic of the next test; since we have no use case, just detect, warn and exit
            # if isinstance(field.type_, _UnionGenericAlias):  ≥3.9
            if str(field.type_).startswith("typing.Union"):
                if any((isinstance(T, type) and issubclass(T, ValidatingConfigBase))
                       for T in field.type_.__args__):
                    raise TypeError("Using a subclass of `ValidatingConfig` inside a Union is not supported.")
            elif isinstance(field.type_, type) and issubclass(field.type_, ValidatingConfigBase):
                if fieldnm not in cfdict:
                    cfdict[fieldnm] = {}
                else:
                    assert isinstance(cfdict[fieldnm], dict), f"Configuration field '{fieldnm}' should be a dictionary."  # Must be mutable
                subcfdict = cfdict[fieldnm]
                for subfieldnm, subfield in field.type_.__fields__.items():
                    if subfieldnm not in subcfdict and subfieldnm in cfdict:
                        # A field is missing, and a plausible default is available
                        # QUESTION: Should we make a (deep) copy, in cause two validating configs make conflicting changes ?
                        subcfdict[subfieldnm] = cfdict[subfieldnm]

                # A ValidatingConfig type requires `rootdir`
                if "rootdir" not in cfdict[fieldnm]:
                    cfdict[fieldnm]["rootdir"] = rootdir

        super().__init__(rootdir=rootdir, **cfdict)

        # # Singleton pattern
        # ValidatingConfigBase.__instances[type(self)] = self

    def __dir__(self):
        return list(self.__fields__)

    @validator("*", pre=True)
    def pass_rootdir(cls, val, values, field):
        "Pass the value of `rootdir` to nested subclasses"
        if lenient_issubclass(field.type_, ValidatingConfigBase):
            cur_rootdir = values.get("rootdir")
            if isinstance(val, dict) and "rootdir" not in val:
                # TODO: Any other types of arguments we should support ?
                val["rootdir"] = cur_rootdir
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

class ValidatingConfig(ValidatingConfigBase, metaclass=ValidatingConfigMeta):
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

    `ValidatingConfig` should be imported and instantiated from within
    ``MyPckg.config.__init__.py``::

       from pathlib import Path
       from mackelab_toolbox.config import ValidatingConfig

       class Config(ValidatingConfig):
           arg1: <type>
           arg2: <type>
           ...

       config = Config(Path(__file__).parent/".project-defaults.cfg",
                       config_module_name=__name__)

    `project.cfg` should be excluded by `.gitignore`. This is where users can
    modify values for their local setup. If it does not exist, a template one
    is created from the contents of `.project-defaults.cfg`, along with
    instructions.

    There are some differences and magic behaviours compared to a plain
    BaseModel, which help to reduce boilerplate when defining configuration options:
    - Defaults are validated (`validate_all = True`).
    - Values of ``"<default>"`` are replaced by the hard coded default in the Config
      definition. (These defaults may be `None`.)
    - Nested plain classes are automatically converted to inherit ValidatingConfigBase,
      and a new attribute of that class type is created. Specifically, if we
      have the following:

          class Config(ValidatingConfig):
              class paths:
                  projectdir: Path

      then this is automatically converted to

          class Config(ValidatingConfig):
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
    - If a user configuration file is not found, and the argument
      `ensure_user_config_exists` is `True`, then a new blank configuration file
      is created at the location we would have expected to find one: inside
      the nearest parent directory which is a version control repository.
    - The `rootdir` value is the path of the directory containing the user config.
    - All arguments of type `Path` are made absolute by prepending `rootdir`.
      Unless they already are absolute, or the class variable
      `make_paths_absolute` is set to `False`.
      !! NOT CURRENTLY TRUE !!: For reasons I don’t yet understandand, the
      mechanism to do this adds the correct validator, but it isn’t executed.
      For the moment, please import the `prepend_rootdir` function and apply it
      (wrapping with ``validator(field)(prepend_rootdir)`` to all relevant fields.
    - A `rootdir` field is added to all auto-generated `ValidatingConfigBase`
      nested classes, and its default value is set to that of the parent.

    Inside config/__init__.py, one would have:

        config = Config(Path(__file__)/".project-defaults.cfg" 
    """
    # rootdir: Path

    package_name: ClassVar[str]
    default_config_file: ClassVar[Union[str,Path]]
    ensure_user_config_exists: ClassVar[bool]=False  # Set to True to add a template config file when none is found

    #path_default_config: Path="config/.project-defaults.cfg"  # Rel path => prepended with rootdir
    #path_user_config: Path="../project.cfg"  # Rel path => prepended with rootdir
    user_config_filename: ClassVar[str]="project.cfg"  # Searched for, starting from CWD

    top_message_default: ClassVar = """
        # This configuration file for '{package_name}' should be excluded from
        # git, so can be used to configure machine-specific variables.
        # This can be used for example to set output paths for figures, or to
        # set flags (e.g. using GPU or not).
        # Default values are listed below; uncomment and edit as needed.
        #
        # Adding a new config field is done by modifying the config module `{config_module_name}`.
        
        """

    # NB: It would be nicer to use Pydantic mechanisms to deal with the defaults for
    #     path arguments. But that would require also implementing `ensure_user_config`
    #     as a validator – not sure that would actually be simpler
    # TODO: Move all arguments to class variables; match signature of ValidatingConfigBase
    def __init__(self,
                 # default_config_file: Union[None,str,Path]=None,
                 cwd: Union[None,str,Path]=None,
                 # ensure_user_config_exists: bool=False,
                 # rootdir: Union[str,Path],
                 # path_default_config=None, path_user_config=None,
                 *,
                 interpolation=None,
                 empty_lines_in_values=False,
                 config_module_name: Optional[str]=None,
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


        See also `ValidatingConfig.ensure_user_config_exists`.
        
        Parameters
        ----------
        default_config_file: Path to the config file used for defaults.
            SPECIAL CASE: If this value is None, then `ValidatingConfig`
            behaves like `ValidatingConfigBase`: no config file is parsed or
            created. This is intended for including as a component of a larger
            ValidatingConfig.
            The assumption then is that all fields are passed by keywords.
            NOT TRUE: Currently we do the special case if kwargs is non-empty.
        cwd: "Current working directory". The search for a config file starts
            from this directory then walks through the parents. 
            The value of `None` indicates to use the current working directory;
            especially if running on a local machine, this is generally what
            you want, and is usually the most portable.
        ensure_user_config_exists: Set to true if you want a template config
            file to be created in a standard location, if no user config file
            is found.
        *
        interpolation: Passed as argument to ConfigParser. Default is
            ExtendedInterpolation(). (Note that, as with ConfigParser, an
            *instance* must be passed.)
        empty_lines_in_values: Passed as argument to ConfigParser. Default is
            True: this prevents multiline values with empty lines, but makes
            it much easier to indent without accidentally concatenating values.
        config_module_name: The value of __name__ when called within the
            project’s configuration module.
            Used for autogenerated instructions in the template user config file.

        **kwargs:
            Additional keyword arguments are passed to ConfigParser.

        Todo
        ----
        When multiple config files are present in the hierarchy, combine them.
        Relative paths in lower config files should still work.
        """
        # HACK !! : Support for (re)-instantiating a config is ill-defined
        #           This way way support updating fields is more accidental than intended, so may be fragile
        if getattr(self, "rootdir", False):  # Hacky way to check if __init__ has already been run on this object
            # Already initialized; if there are new kwargs, validate them.
            for field, val in kwargs.items():
                setattr(self, field, val)  # Relies on `validate_assignment = True`
        elif kwargs:
            # We have NOT yet initialized, but are passing keyword arguments:
            # this may happen because of __new__ returning an existing instance,
            # in which case this __init__ gets executed twice
            # (self.dict() should be empty, but in general we would do this to avoid overwriting an already set value)
            super().__init__(**{**self.dict(),**kwargs})
        else:  # Normal path
            ## Search for `project.cfg` file in the current directory and its parents ##
            # If no project config is found, create one in the location documented above
            if cwd is None:
                cwd = Path(os.getcwd())
            default_location_for_conffile = None  # Will be set the first time we find a .git folder
            conffile = self.user_config_filename
           
            wdfiles = set(os.listdir(cwd))
            if conffile in wdfiles:
                rootdir = cwd
            else:
                rootdir = None
                if {".git", ".hg", ".svn"} & wdfiles:
                    default_location_for_conffile = cwd/conffile
                for wd in cwd.parents:
                    wdfiles = set(os.listdir(wd))
                    if conffile in wdfiles:
                        rootdir = wd
                        # break
                    elif ({".git", ".hg", ".svn"} & wdfiles
                          and not default_location_for_conffile):
                        default_location_for_conffile = wd/conffile

                if rootdir is None and self.ensure_user_config_exists:
                    logger.warning(f"Could not find a file '{conffile}' in '{cwd}' or its parents.")

            # Read the config file(s)
            if interpolation is None: interpolation = ExtendedInterpolation() 
            cfp = ConfigParser(interpolation=interpolation,
                               empty_lines_in_values=empty_lines_in_values,
                               **kwargs)
            with open(self.default_config_file) as f:
                cfp.read_file(f)

            if rootdir:
                cfp.read(rootdir/conffile)
            elif default_location_for_conffile is not None:
                if self.ensure_user_config_exists:
                    # We didn’t find a project config file, but we did find that
                    # we are inside a VC repo => place config file at root of repo
                    # `ensure_user_config_exists` creates a config file from the
                    # defaults file, listing all default values (behind comments),
                    # and adds basic instructions and the default option values
                    self.add_user_config_if_missing(
                        self.default_config_file, default_location_for_conffile,
                        self.package_name, config_module_name)
            elif self.ensure_user_config_exists:
                logger.error(f"The provided current working directory ('{cwd}') "
                             "is not part of a version controlled repository.")

            # Convert cfp to a dict; this loses the 'defaults' functionality, but makes
            # it much easier to support validation and nested levels
            cfdict = {section: dict(values) for section, values in cfp.items()}

            # Use Pydantic to validate the values read into `cfp`
            super().__init__(rootdir=rootdir, **cfdict)

    def add_user_config_if_missing(
        self,
        path_default_config: Union[str,Path],
        path_user_config: Union[str,Path],
        package_name: str,
        config_module_name: str="config",
        ):
        """
        If the user-editable config file does not exist, create it.

        Basic instructions are added as a comment to the top of the file.
        Their content is determined by the class variable `top_message_default`.
        Two variables are available for substitution into this message:
        `package_name` and `config_module_name`.

        Parameters
        ----------
        path_default_config: Path to the config file providing defaults.
            *Should* be version-controlled
        path_user_config: Path to the config file a user would modify.
            Should *not* be version-controlled
        path_config_module: Name of the python module defining config fields.
            Only used for the top message.
        package_name: The name of the package using the config object.
            Only used for the top message.
        config_module_name: String which may optionally used for substitution
            in `self.top_message_default`.
        """
        top_message = self.top_message_default
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

def prepend_rootdir(val, values):
    if isinstance(val, Path) and not val.is_absolute():
        rootdir = values.get("rootdir")
        if rootdir:
            val = rootdir/val
    return val

def ensure_dir_exists(cls, dirpath):
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    return dirpath
