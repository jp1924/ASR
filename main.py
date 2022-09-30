import os
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config, HfArgumentParser, Trainer
from transformers.integrations import WandbCallback
from datasets import load_dataset
from setproctitle import setproctitle


def main() -> None:
    cache = r"/data/jsb193/github/t5/.cache"
    model_name_or_path = "KETI-AIR/ke-t5-base"

    tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path, cache_dir=cache)
    config = T5Config.from_pretrained(model_name_or_path, cache_dir=cache)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=config, cache_dir=cache)

    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else None
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    return


if "__main__" == __name__:
    # check wandb env variable
    # https://docs.wandb.ai/guides/track/advanced/environment-variables
    # parser = HfArgumentParser()
    process_name = "[JP]T5_test"
    setproctitle(process_name)
    main()
