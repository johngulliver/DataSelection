from DataSelection.configs.config_node import ConfigNode

config = ConfigNode()

# data selector
config.selector = ConfigNode()
config.selector.type = None
config.selector.model_name = None
config.selector.model_config_path = None
config.selector.training_dynamics_data_path = None
config.selector.burnout_period = 0
config.selector.number_samples_to_relabel = 10

# tensorboard
config.tensorboard = ConfigNode()
config.tensorboard.save_events = True

def get_default_selector_config() -> ConfigNode:
    return config.clone()
