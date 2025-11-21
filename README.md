# Scaling Feedback Descent

This project implements a "Critic" model using the FeedSum dataset and the Tinker framework.

## Setup

### Prerequisites

- Python 3.12+
- [Tinker](https://github.com/thinking-machines-lab/tinker) API Key

### Installation

1. Create a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Note**: `tinker-cookbook` is included in `requirements.txt`. If you wish to install it from the local directory in editable mode (useful for development), run:
   > ```bash
   > pip install -e tinker-cookbook/
   > ```

## Usage

### Training the Critic

Run the training script:

```bash
python train_critic.py
```

## Project Structure

- `prepare_data.py`: Script to download and prepare the FeedSum dataset.
- `train_critic.py`: Main script to train the critic model.
- `run_pipeline.py`: Script to run the evaluation pipeline.
- `tinker-cookbook/`: Submodule containing Tinker recipes.