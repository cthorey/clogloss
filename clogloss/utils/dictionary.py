import collections


def flatten(d, parent_key='', sep='__', ignored_keys=[]):
    """
    Flatten dictionary

    Example:

    d = {'mgc_config': {'common': {'set_num_planning_attempts': 4,
    'set_planner_id': 'RRTConnectkConfigDefault'}}}
    pprint(self.flatten(d,inverse=False,sep="__"))
    {'mgc_config__common__set_num_planning_attempts': 4,
    'mgc_config__common__set_planner_id': 'RRTConnectkConfigDefault'}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping) \
                and k not in ignored_keys:
            items.extend(
                flatten(v, new_key, sep=sep,
                        ignored_keys=ignored_keys).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten(d, sep='__'):
    """
    UnFlatten dictionary
    Example:
    d = {'mgc_config__common__set_num_planning_attempts': 4,
    'mgc_config__common__set_planner_id': 'RRTConnectkConfigDefault'}

    pprint(self.flatten(d,inverse=True,sep="__"))
    {'mgc_config': {'common': {'set_num_planning_attempts': 4,
    'set_planner_id': 'RRTConnectkConfigDefault'}}}
    """
    out = dict()
    for key, val in d.items():
        parent_key = key.split(sep)[0]
        if len(key.split(sep)) == 1:
            out[parent_key] = val
        else:
            d0 = {
                sep.join(k.split(sep)[1:]): v
                for k, v in d.items() if k.split(sep)[0] == parent_key
            }
            out[key.split(sep)[0]] = unflatten(d0, sep=sep)
    return out
