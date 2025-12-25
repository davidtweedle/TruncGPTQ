import torch
import gc
import utils
import data_utils
import model_utils
import eval_utils
from gptq_utils import gptq_svd_fwrd, Quantizer


def main():
    print(f"Starting quantization")
    args = utils.get_args()
    torch.manual_seed(args.seed)
    model, tokenizer = model_utils.get_model(args.model_id, args.device)
    input_ids_list = data_utils.get_loaders(args.dataset, tokenizer, args.n_samples, args.seq_len)

    inps, outs, attention_mask, position_ids = model_utils.capture_initial_inputs(
            model, input_ids_list, args.device
            )
    layers = model_utils.get_layers(model)

    layer_inputs = {}
    def add_batch(name):
        def hook(module, input, output):
            inp = input[0].detach()
            if len(inp.shape) == 3:
                inp = inp.squeeze(0)
            if name not in layer_inputs:
                layer_inputs[name] = []
            layer_inputs[name].append(inp)
        return hook

    for i, layer in enumerate(layers):
        layer = layer.to(args.device)

        subset = model_utils.find_linear_layers(layer)
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.n_samples):
            with torch.no_grad():
                layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids
                        )
        for h in handles:
            h.remove()

        for name, submodule in subset.items():
            if name not in layer_inputs or len(layer_inputs[name]) == 0:
                print(f"Warning: No inputs captured for {name}, skipping quantization")
                # should round to nearest
                continue
            X_list = layer_inputs[name]
            W = submodule.weight.data.float()
            m, n = W.shape
            sketch_dim = int(n * args.sketch_ratio)

            def make_stream_adapter():
                for x_chunk in X_list:
                    yield x_chunk.to(torch.float32)
            
            out_weight = torch.zeros_like(W)
            quantizer = Quantizer(per_channel=True, w_bits=args.w_bits)

            gptq_svd_fwrd(
                    sketch_dim=sketch_dim,
                    oversample=16,
                    k_iter=args.k_iter,
                    make_stream=make_stream_adapter,
                    weight_mat=W,
                    out_weight=out_weight,
                    quantizer=quantizer,
                    eps=args.eps
                    )

            submodule.weight.copy_(out_weight)

            del X_list, layer_inputs[name]
            gc.collect()
        for j in range(args.n_samples):
            with torch.no_grad():
                outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids
                        )[0]
        inps, outs = outs, inps
        torch.cuda.empty_cache()

        print(f"Saving model to {args.save_path}...")
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)

        eval_utils.evaluate_perplexity(model, tokenizer, device=args.device)

    if __name__ == "__main__":
        main()
