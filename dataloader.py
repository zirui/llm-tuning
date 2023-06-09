import transformers
from torch.utils.data import Dataset
import json
import logging

PROMPT_DICT = {
  "prompt_input": ("{instruction}\n\n {input}\n\n"),
  "prompt_no_input": ("{instruction}\n\n"),
}


class Seq2SeqDataset(Dataset):

  def __init__(self, data_path):
    super(Seq2SeqDataset, self).__init__()
    logging.warning("Loading data...")
    with open(data_path, "r") as f:
      list_data_dict = json.load(f)

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT[
      "prompt_no_input"]

    logging.warning("Formatting data...")
    sources = [
      prompt_input.format_map(example) if example.get("input", "") != "" else
      prompt_no_input.format_map(example) for example in list_data_dict
    ]
    targets = [f"{example['output']}" for example in list_data_dict]

    self.sources = sources
    self.targets = targets

  def __len__(self):
    return len(self.sources)

  def __getitem__(self, item):
    return self.sources[item], self.targets[item]


class Seq2SeqCollator(object):

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer

  def __call__(self, batch):
    sources = [ex[0] for ex in batch]
    targets = [ex[1] for ex in batch]

    inputs = self.tokenizer(sources,
                            max_length=40,
                            return_tensors='pt',
                            padding=True,
                            truncation=True)

    labels = self.tokenizer(targets,
                            max_length=160,
                            return_tensors='pt',
                            padding=True,
                            truncation=True).input_ids

    inputs['labels'] = labels

    return inputs
