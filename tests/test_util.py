import inspect

from cog import BasePredictor
import pydantic


def get_params(p: BasePredictor, passed_params: dict):
    sig = inspect.signature(p.predict)
    defaults = {
        name: param.default.default if isinstance(param.default, pydantic.fields.FieldInfo) else param.default
        for name, param in sig.parameters.items()
        if name not in passed_params and name != 'self'
    }
    
    return {**defaults, **passed_params}