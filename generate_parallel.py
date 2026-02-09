import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens




@torch.no_grad()
def generate_parallel(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128, 
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    semantic_block_mask_lengths: list[int] | None = None,
):
    """
    Parallel (synchronous) block update generate.
    semantic_block_mask_lengths: if provided, list of mask lengths per semantic block (must sum to gen_length).
        Example: [128,128,128] for 3 steps each with 128 masks.
    If semantic_block_mask_lengths is None, we fall back to splitting gen_length into equal blocks by block_length.
    """
    


    device = model.device
    prompt = prompt.to(device)
    if prompt.dim() != 2 or prompt.shape[0] != 1:
        raise ValueError("prompt must be shape (1, L)")

    x = prompt.clone()
    total_len = x.shape[1]
    
    #print("DBG: prompt shape:", x.shape)
    #print("DBG: prompt token ids (first 100):", x[0,:100].tolist())
    #print("DBG: total mask_count in prompt:", int((x == mask_id).sum().item()))



    prompt_index = (x != mask_id)  
    
    #print("DBG: mask_id:", mask_id, "prompt_index.sum():", int(prompt_index.sum().item()))

    if semantic_block_mask_lengths is None:
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length if no semantic lengths given"
        num_blocks = gen_length // block_length
    else:
        num_blocks = len(semantic_block_mask_lengths) 

    block_ranges = []
    
    #print("DEBUG: total mask count in prompt:", (prompt == mask_id).sum().item())
    #print("DEBUG: semantic_block_mask_lengths:", semantic_block_mask_lengths)
    
    offset = 0
    for L in semantic_block_mask_lengths:
        start = offset
        end = L + offset 
        block_ranges.append((start, end))
        offset = end
    # Build boolean mask per block; shape (num_blocks, total_len)
    total_len = x.shape[1] 
    block_masks = torch.zeros((num_blocks, total_len), dtype=torch.bool, device=device)
    for b, (s, e) in enumerate(block_ranges):
        if(e < total_len-1):
            block_masks[b, e-block_length : e] = True
        else:
            block_masks[b, e- 2* block_length:e] = True

    per_block_num_transfer = []
    for b in range(num_blocks):
        mask_idx = block_masks[b:b+1, :]  # shape (1, L)
        per_block_num_transfer.append(get_num_transfer_tokens(mask_idx, steps))  
    
    #print("DEBUG: per_block_num_transfer shape:", len(per_block_num_transfer))
    ##print("DEBUG: per_block_num_transfer sample:", per_block_num_transfer[:, :min(10, per_block_num_transfer.shape[1])])
    

        
    per_block_num_transfer = torch.cat(per_block_num_transfer, dim=0).squeeze(1)  # (num_blocks, steps)


    for i in range(steps):
        ##print(f"STEP {i}: per_block_num_transfer = {per_block_num_transfer.tolist()}")
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)  
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits  

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # (batch, L)

        if remasking == 'low_confidence':
            p = F.softmax(logits.to(torch.float32), dim=-1)  # (batch, L, V)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # (batch, L)

        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)
  
        mask_index_all = (x == mask_id)  # (batch, L)
        
        #_, cols = torch.nonzero(mask_index_all, as_tuple=True)
        #mask_positions = cols.tolist()
        ##print("mask_positions:", mask_positions, "num_masks:",len(mask_positions))
        
        
        ##print(f"DBG STEP {i}: mask_index_all.sum()={mask_index_all.sum().item()}")
        ##print("DEBUG: x0_p[0,:20] =", x0_p[0,:20])
        
        x0 = torch.where(mask_index_all, x0, x)  
        
        #confidence = torch.where(mask_index_all, x0_p, torch.tensor(-np.inf, device=x.device, dtype=x0_p.dtype))
        ##print("DBG: x.device, x0_p.device, x.dtype, x0_p.dtype", x.device, x0_p.device, x.dtype, x0_p.dtype)
        ##print("DBG: mask_id in x?", (x == mask_id).any())
        ##print("mask_index_all.device:", mask_index_all.device, "mask_index_all.dtype:", mask_index_all.dtype)


        confidence = torch.full(x0_p.shape, -np.inf, device=x0.device, dtype=x0_p.dtype)
        
        confidence = confidence.to(x0.device)
        confidence = confidence.masked_scatter(mask_index_all, x0_p[mask_index_all])
        

        ##print("confidence[0,:20]:", confidence[0,:20])
        ##print("confidence non-inf count:", (confidence > -1e30).sum().item())



        # Build transfer_index initially False
        transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        
        
        total_transferred = int(transfer_index.sum().item())
        ##print(f"DBG STEP {i}: total_transferred={total_transferred}")


        for b in range(num_blocks):
            # positions in this block that are currently mask, actually refer to semantic block length
            block_mask = block_masks[b:b+1, :].clone()  # (1, L)
            ###print(f"block_mask shape:{block_mask.shape}")
            #_, cols = torch.nonzero(block_masks, as_tuple = True)
            #now_mask = cols.tolist()
            ###print("now_mask:", now_mask, "num_in_now mask:", len(now_mask))
            
            
            block_transferred = int((transfer_index & block_mask).sum().item())
            ##print(f"  block {b} transferred this step: {block_transferred}, scheduled k: {int(per_block_num_transfer[b, i].item())}")
            # consider only mask positions (both block and overall mask)
            candidate_idx = (mask_index_all & block_mask)  # (batch, L)
            ##print(f"DBG STEP {i}: block {b} candidate_idx.sum()={candidate_idx.sum().item()}")

            if candidate_idx.sum() == 0:
                continue
            k = int(per_block_num_transfer[b, i].item())

            if k <= 0:
                continue
            conf_vec = confidence.clone()  # (batch, L)
            conf_vec = torch.where(candidate_idx, conf_vec, torch.tensor(-np.inf, device=conf_vec.device))
            for j in range(conf_vec.shape[0]):
                avail = int((candidate_idx[j]).sum().item())
                if avail == 0:
                    continue
                kk = min(k, avail)
                _, sel_pos = torch.topk(conf_vec[j], k=kk)
                transfer_index[j, sel_pos] = True

        x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    #print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()