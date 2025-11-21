from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
import asyncio
import os

def build_config() -> train.Config:
    # Model configuration
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    
    # Dataset configuration
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=4,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    
    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path="data/pairwise_train.jsonl",
    )
    
    return train.Config(
        log_path="logs/sft_feedsum_critic_pairwise",
        model_name=model_name,
        dataset_builder=dataset,
        wandb_project="Scaling_Feedback_Descent",
        wandb_name="sft_feedsum_critic_pairwise_qwen3_4b",
        num_epochs=1,
        save_every=100,
        eval_every=100,
    )

def main():
    load_dotenv()
    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))

if __name__ == "__main__":
    main()
