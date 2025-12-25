import torch
from torch import nn
from tqdm import tqdm


def evaluate_perplexity(model, tokenizer, dataset="wikitext2", device="cuda"):
    if dataset == "wikitext2":
        from datasets import load_dataset
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(testdata["text"])
    else:
        return -1
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.seqlen
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    model.eval()
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\nPerplexity: {ppl.item():.4f}")
    return ppl.item()
