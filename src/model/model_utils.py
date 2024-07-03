from transformers import AutoConfig


def auto_upgrade(config):
    # TODO: check later
    cfg = AutoConfig.from_pretrained(config)
    if 'aligngpt' in config and 'aligngpt' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer AlignGPT code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "aligngpt")
            cfg.architectures[0] = 'AlignGPTForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
