import os
import argparse
import logging
from typing import Iterator

import chz
import datasets
import tinker
from tinker_cookbook.preference.train_dpo import Config as DPOConfig, main as dpo_main
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)

@chz.chz
class DPOFeedbackDatasetBuilder(ChatDatasetBuilder):
    """
    Custom Dataset Builder for the interleaved DPO dataset from HF.
    Expects dataset rows like:
    row 0: Chosen (winner)
    row 1: Rejected (loser)
    """

    hf_repo_id: str
    split: str = "train"

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        dataset = datasets.load_dataset(self.hf_repo_id, split=self.split)
        if len(dataset) % 2 != 0:
            raise ValueError("Dataset length must be even (interleaved chosen/rejected pairs).")
        
        def group_pairs(examples):
            chosen_msgs = examples["messages"][0::2]
            rejected_msgs = examples["messages"][1::2]
            return {
                "chosen_messages": chosen_msgs,
                "rejected_messages": rejected_msgs
            }

        paired_dataset = dataset.map(
            group_pairs,
            batched=True,
            batch_size=1000, 
            remove_columns=dataset.column_names
        )

        renderer = self.renderer
        max_length = self.common_config.max_length

        def example_to_data(example: dict) -> list[tinker.Datum]:
            chosen_msgs = example["chosen_messages"]
            rejected_msgs = example["rejected_messages"]

            chosen_tokens, chosen_weights = renderer.build_supervised_example(chosen_msgs)
            rejected_tokens, rejected_weights = renderer.build_supervised_example(rejected_msgs)

            return [
                datum_from_model_input_weights(chosen_tokens, chosen_weights, max_length),
                datum_from_model_input_weights(rejected_tokens, rejected_weights, max_length),
            ]

        supervised_dataset = SupervisedDatasetFromHFDataset(
            paired_dataset,
            batch_size=self.common_config.batch_size,
            flatmap_fn=example_to_data,
        )

        return supervised_dataset, None

def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train Feedback Model with DPO")
    parser.add_argument("--repo_id", type=str, default="JasonYan777/dpo_data_v1", help="Hugging Face repo ID")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Base model for training")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (number of pairs per step)")
    parser.add_argument("--log_dir", type=str, default="~/tinker_logs/dpo_feedback", help="Logging directory")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO Beta parameter")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA Rank for training")
    
    args = parser.parse_args()

    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    from tinker_cookbook.model_info import get_recommended_renderer_name

    raw_model_name = args.model.replace("tinker/", "") if args.model.startswith("tinker/") else args.model

    dataset_builder = DPOFeedbackDatasetBuilder(
        hf_repo_id=args.repo_id,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=raw_model_name,
            renderer_name=get_recommended_renderer_name(raw_model_name),
            batch_size=args.batch_size,
            max_length=4096
        )
    )

    config = DPOConfig(
        log_path=args.log_dir,
        model_name=raw_model_name,
        dataset_builder=dataset_builder,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        dpo_beta=args.dpo_beta,
        lora_rank=args.lora_rank,
        eval_every=0, # disable eval
        save_every=100,
        wandb_project="scaling_feedback_descent",
        wandb_name="feedback_dpo_training"
    )

    dpo_main(config)

if __name__ == "__main__":
    main()
