from .hotpotqa import HotpotQATask
from .gsm8k import GSM8kTask
from .strategyqa import StrategyQATask
from .mmlu import MMLUTask
from .svamp import SVAMPTask
from .bbh import BBHTask
from .bamgoogle import BamGoogleTask


def get_task(name, split):
    if name == "hotpotqa":
        return HotpotQATask(split)
    elif name == "gsm8k":
        return GSM8kTask(split)
    elif name == "strategyqa":
        return StrategyQATask(split)
    elif name == "mmlu":
        return MMLUTask(split)
    elif name == "svamp":
        return SVAMPTask(split)
    elif name == "bbh":
        return BBHTask(split)
    elif name == "bamgoogle":
        return BamGoogleTask(split)
    else:
        raise ValueError("Unknown task name: {}".format(name))
    
