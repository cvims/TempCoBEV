from opencood.models.temporal_modules.identity import IdentityEncoder
from opencood.models.temporal_modules.tempcobev import TempCoBEV


__all__ = dict(
    IdentityEncoder=IdentityEncoder,
    TempCoBEV=TempCoBEV
)

def build_temporal_module(cfg):
    module_name = cfg['core_method']
    error_message = f"{module_name} is not found. " \
                    f"Please add your temporal module file's name in opencood/" \
                    f"models/temporal_modules/init.py"

    # get fusion level modules
    registered_modules = __all__

    assert module_name in registered_modules, error_message

    kwargs = cfg['args']
    if kwargs == []:
        kwargs = {}

    _module = registered_modules[module_name](
        **kwargs
    )

    return _module
