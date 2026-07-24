"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.

    token_to_id = {}

    for i, token in enumerate(specials):
          token_to_id[token] = i

    next_id = len(specials)
    for sentence in sentences:
        for token in sentence.split():
            if token not in token_to_id:
                token_to_id[token] = next_id
                next_id += 1
    return token_to_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    id_to_token = {}
    for token, tid in token_to_id.items():
        id_to_token[tid] = token
    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    ids = []
    unk_id = token_to_id[unk_token]  # 先拿到 <unk> 的 id，免得每次循环都查

    for token in sentence.split():
        if token in token_to_id:
            ids.append(token_to_id[token])
        else:
            ids.append(unk_id)  # 不认识的词，统一用 <unk> 兜底

    return ids

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    res = []
    for tid in ids:
        res.append(id_to_token[tid])
    return res

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    if len(ids) < max_len:
        return ids + [pad_id] * (max_len-len(ids))
    else:
        return ids[:max_len]

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.tensor(padded_sequences,dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings*math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    half = d_model // 2
    i = torch.tensor([idx for idx in range(half)], dtype=torch.float32)
    return 1.0 / (10000.0 ** (2 * i / d_model))

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(max_len,dtype=torch.float32).unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pe[:,0::2] = torch.sin(position*div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe[:,1::2] = torch.cos(position*div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros(max_len,d_model,dtype=torch.float32)
    div_term = compute_positional_div_term(d_model)
    position =  build_position_index_column(max_len)
    pe = fill_even_indices_with_sin(pe,position,div_term)
    pe = fill_odd_indices_with_cos(pe,position,div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    return embedded_batch + positional_encoding[:embedded_batch.size(1)].unsqueeze(0)

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    mask = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    return query @ key.transpose(-2,-1)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(~mask, float("-inf"))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    weights = torch.nn.functional.softmax(masked_scores,dim=-1)
    return torch.nan_to_num(weights)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    scores = compute_raw_attention_scores(query,key)
    d_k = query.shape[-1]
    scores = scale_attention_scores(scores,d_k)

    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores,mask)
    
    weights = softmax_attention_weights(scores)
    context = apply_attention_weights_to_values(weights,value)
    
    return context,weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B,L,d_model = tensor.shape
    d_k = d_model // num_heads
    return tensor.reshape(B,L,num_heads,d_k)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1,2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B,H,L,d_k = multi_head_tensor.shape
    return multi_head_tensor.transpose(1,2).reshape(B,L,H*d_k).contiguous()

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is None:
        return x @ weight.T
    else:
        return x @ weight.T + bias

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    q = apply_linear_projection(x,w_q,b_q)
    k = apply_linear_projection(x,w_k,b_k)
    v = apply_linear_projection(x,w_v,b_v)
    return q,k,v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))
    return q_h,k_h,v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h,k_h,v_h,mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    merged = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(merged,w_o,b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    q = apply_linear_projection(query,w_q,None)
    k = apply_linear_projection(key,w_k,None)
    v = apply_linear_projection(value,w_v,None)

    q_h,k_h,v_h = split_qkv_into_heads(q, k, v, num_heads)

    context,weights = multi_head_scaled_dot_product_attention(q_h,k_h,v_h,mask)

    output = merge_heads_and_project_output(context,w_o,None)

    return output

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    hidden = apply_ffn_first_linear_and_relu(x,w1,b1)
    return apply_ffn_second_linear(hidden,w2,b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = x.mean(dim = -1,keepdim = True)
    var = ((x-mean)**2).mean(dim=-1,keepdim=True)
    return mean,var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean,var = compute_layer_norm_mean_and_variance(x)
    x_norm = (x-mean)/torch.sqrt(var+eps)
    return gamma * x_norm + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    combined = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(combined,gamma,beta,eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    return x * keep_mask / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    attn_out = assemble_multi_head_attention_forward(x,x,x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(x,attn_out,gamma,beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    ffn_out = position_wise_feed_forward_network(x, w1, b1, w2, b2,)
    return apply_residual_add_and_norm(x,ffn_out,gamma,beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    x = encoder_layer_self_attention_sublayer(
        x, layer_params["w_q"], layer_params["w_k"], layer_params["w_v"],
        layer_params["w_o"], layer_params["attn_gamma"], layer_params["attn_beta"],
        num_heads, src_mask
    )
    x = encoder_layer_feed_forward_sublayer(
        x, layer_params["w1"], layer_params["b1"], layer_params["w2"],
        layer_params["b2"], layer_params["ffn_gamma"], layer_params["ffn_beta"]
    )
    return x

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    for layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(x, layer_params, num_heads, src_mask)
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    attn_out = assemble_multi_head_attention_forward(y,y,y,w_q, w_k, w_v,w_o,num_heads,tgt_mask)
    return apply_residual_add_and_norm(y,attn_out,gamma,beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    attn_out = assemble_multi_head_attention_forward(y,encoder_output,encoder_output,w_q,w_k,w_v,w_o,num_heads,src_mask)
    return apply_residual_add_and_norm(y,attn_out,gamma,beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    ffn = position_wise_feed_forward_network(y,w1,b1,w2,b2,)
    return apply_residual_add_and_norm(y,ffn,gamma,beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # TODO: chain the three decoder sublayers using params from layer_params.
    y = decoder_layer_masked_self_attention_sublayer(
        y,
        layer_params["w_q_self"], layer_params["w_k_self"],
        layer_params["w_v_self"], layer_params["w_o_self"],
        layer_params["self_gamma"], layer_params["self_beta"],
        num_heads, tgt_mask
    )
    y = decoder_layer_cross_attention_sublayer(
        y, encoder_output,
        layer_params["w_q_cross"], layer_params["w_k_cross"],
        layer_params["w_v_cross"], layer_params["w_o_cross"],
        layer_params["cross_gamma"], layer_params["cross_beta"],
        num_heads, src_mask
    )
    y = decoder_layer_feed_forward_sublayer(
        y,
        layer_params["w1"], layer_params["b1"],
        layer_params["w2"], layer_params["b2"],
        layer_params["ffn_gamma"], layer_params["ffn_beta"]
    )
    return y

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    for layer_params in decoder_layer_params_list:
        y = assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask)
    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return torch.nn.functional.log_softmax(logits,dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    d_model = model_params["token_embedding"].shape[-1]
    max_len = max(src_ids.shape[1], tgt_ids.shape[1])

    # 1. Embedding + 缩放 + 位置编码（源语言）
    src_emb = model_params["token_embedding"][src_ids]
    src_emb = scale_embeddings_by_sqrt_d_model(src_emb, d_model)
    src_pe = build_sinusoidal_positional_encoding(max_len, d_model)
    src_emb = add_positional_encoding_to_embeddings(src_emb, src_pe)

    # 2. Embedding + 缩放 + 位置编码（目标语言）
    tgt_emb = model_params["token_embedding"][tgt_ids]
    tgt_emb = scale_embeddings_by_sqrt_d_model(tgt_emb, d_model)
    tgt_pe = build_sinusoidal_positional_encoding(max_len, d_model)
    tgt_emb = add_positional_encoding_to_embeddings(tgt_emb, tgt_pe)

    # 3. 构建 mask
    src_mask = build_padding_mask(src_ids, pad_id)
    tgt_pad_mask = build_padding_mask(tgt_ids, pad_id)
    causal_mask = build_causal_mask(tgt_ids.shape[1])
    tgt_mask = combine_padding_and_causal_masks(tgt_pad_mask, causal_mask)

    # 4. 编码器
    encoder_output = stack_encoder_layers(
        src_emb, model_params["encoder_layers"], num_heads, src_mask
    )

    # 5. 解码器
    decoder_output = stack_decoder_layers(
        tgt_emb, encoder_output, model_params["decoder_layers"],
        num_heads, src_mask, tgt_mask
    )

    # 6. 输出投影 → logits → log_probs
    logits = apply_final_output_projection(
        decoder_output, model_params["output_projection"]
    )
    log_probs = apply_log_softmax_over_vocab(logits)

    return log_probs

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    scale = 1.0 / math.sqrt(d_model)
    params = {}
    for name, shape in [
        ("w_q", (d_model, d_model)), ("w_k", (d_model, d_model)),
        ("w_v", (d_model, d_model)), ("w_o", (d_model, d_model)),
        ("w1", (d_model, d_ff)), ("w2", (d_ff, d_model)),
    ]:
        t = torch.empty(shape, dtype=torch.float32)
        t.normal_(0, scale) # 直接乘法和改变分布是等效的操作，乘法会让叶子节点失效
        t.requires_grad_(True)
        params[name] = t

    for name, size in [("b1", d_ff), ("b2", d_model),
                        ("attn_beta", d_model), ("ffn_beta", d_model)]:
        params[name] = torch.zeros(size, dtype=torch.float32, requires_grad=True)

    for name, size in [("attn_gamma", d_model), ("ffn_gamma", d_model)]:
        params[name] = torch.ones(size, dtype=torch.float32, requires_grad=True)

    return params

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    scale = 1.0 / math.sqrt(d_model)
    params = {}
    for name, shape in [
        ("w_q_self", (d_model, d_model)), ("w_k_self", (d_model, d_model)),
        ("w_v_self", (d_model, d_model)), ("w_o_self", (d_model, d_model)),
        ("w_q_cross", (d_model, d_model)), ("w_k_cross", (d_model, d_model)),
        ("w_v_cross", (d_model, d_model)), ("w_o_cross", (d_model, d_model)),
        ("w1", (d_model, d_ff)), ("w2", (d_ff, d_model)),
    ]:
        t = torch.empty(shape, dtype=torch.float32)
        t.normal_(0, scale)
        t.requires_grad_(True)
        params[name] = t

    for name, size in [("b1", d_ff), ("b2", d_model),
                        ("self_beta", d_model), ("cross_beta", d_model),
                        ("ffn_beta", d_model)]:
        params[name] = torch.zeros(size, dtype=torch.float32, requires_grad=True)

    for name, size in [("self_gamma", d_model), ("cross_gamma", d_model),
                        ("ffn_gamma", d_model)]:
        params[name] = torch.ones(size, dtype=torch.float32, requires_grad=True)

    return params

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    scale = 1.0 / math.sqrt(d_model)
    src = torch.empty(vocab_size, d_model, dtype=torch.float32)
    src.normal_(0, scale)
    src.requires_grad_(True)

    tgt = torch.empty(vocab_size, d_model, dtype=torch.float32)
    tgt.normal_(0, scale)
    tgt.requires_grad_(True)

    if tie_weights:
        output_proj = tgt  # 共享同一份参数
    else:
        output_proj = torch.empty(vocab_size, d_model, dtype=torch.float32)
        output_proj.normal_(0, scale)
        output_proj.requires_grad_(True)

    return {
        "src_embedding": src,
        "tgt_embedding": tgt,
        "output_projection": output_proj,
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    result = []
    seen = set()

    for layer in encoder_layer_params:
        for t in layer.values():
            if id(t) not in seen:
                seen.add(id(t))
                result.append(t)

    for layer in decoder_layer_params:
        for t in layer.values():
            if id(t) not in seen:
                seen.add(id(t))
                result.append(t)

    for t in embedding_params.values():
        if id(t) not in seen:
            seen.add(id(t))
            result.append(t)

    return result

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    batch_size, tgt_seq = target_ids.shape
    output = torch.empty_like(target_ids)         # 同 shape、同 dtype、同 device
    output[:, 0] = start_token_id                 # 第一列 = START
    output[:, 1:] = target_ids[:, :-1]            # 后面的列 = target 去掉最后一列
    return output

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    step = max(step, 1)  # 防止第 0 步除零
    arg1 = step ** -0.5                           # 衰减部分: 1/√step
    arg2 = step * (warmup_steps ** -1.5)          # 升温部分: step / warmup_steps^1.5
    return float((d_model ** -0.5) * min(arg1, arg2))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    value = epsilon / (vocab_size - 2)
    return torch.full(shape,value,dtype = torch.float32)

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    result = smoothed_distribution.clone()
    result.scatter_(dim=2, index=gold_token_ids.unsqueeze(-1), value=confidence)
    return result

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    result = smoothed_distribution.clone()
    # 列清零：所有位置的 <pad> 槽位归零
    result[:, :, pad_id] = 0
    # 行清零：gold_token_ids 等于 pad_id 的位置整行归零
    pad_mask = (gold_token_ids == pad_id)  # shape: (batch, tgt_seq)
    result[pad_mask] = 0
    return result

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    return -(smoothed_distribution * log_probabilities).sum()+0.0

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    non_pad_count = (gold_token_ids != pad_id).sum()
    if non_pad_count == 0:
        return total_loss
    return total_loss / non_pad_count

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    predictions = log_probabilities.argmax(dim=-1)                    # (batch, tgt_seq)
    correct = (predictions == gold_token_ids)                          # (batch, tgt_seq)
    non_pad_mask = (gold_token_ids != pad_id)                         # (batch, tgt_seq)
    correct_non_pad = correct & non_pad_mask
    correct_count = correct_non_pad.sum()
    non_pad_count = non_pad_mask.sum()
    if non_pad_count == 0:
        return torch.tensor(0.0, dtype=log_probabilities.dtype, device=log_probabilities.device)
    return correct_count / non_pad_count

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    m = [torch.zeros_like(p) for p in parameter_list]
    v = [torch.zeros_like(p) for p in parameter_list]
    t = 0
    return {'m': m, 'v': v, 't': t}

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    return  beta1*m_prev + (1-beta1) * grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    return beta2 * v_prev + (1 - beta2) * grad ** 2

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    m_hat = m_t / (1 - beta1 ** step)
    v_hat = v_t / (1 - beta2 ** step)
    return m_hat, v_hat

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    optimizer_state['t'] += 1
    t = optimizer_state['t']

    for i, param in enumerate(parameter_list):
        if param.grad is None:
            continue

        grad = param.grad

        # 更新一阶动量 m
        optimizer_state['m'][i] = beta1 * optimizer_state['m'][i] + (1 - beta1) * grad

        # 更新二阶动量 v
        optimizer_state['v'][i] = beta2 * optimizer_state['v'][i] + (1 - beta2) * grad ** 2

        # 偏差修正
        m_hat = optimizer_state['m'][i] / (1 - beta1 ** t)
        v_hat = optimizer_state['v'][i] / (1 - beta2 ** t)

        # 计算 delta 并更新参数
        delta = learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
        param.data -= delta

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for param in parameter_list:
        param.grad = None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # TODO: shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    pad_id = config["pad_id"]
    start_id = config["start_id"]
    vocab_size = config["vocab_size"]
    smoothing = config["smoothing"]
    num_heads = config["num_heads"]

    # 补齐 model_params 里可能缺失的 key
    if "token_embedding" not in model_params:
        model_params["token_embedding"] = model_params["src_embedding"]
    if "d_model" not in model_params:
        model_params["d_model"] = model_params["token_embedding"].shape[-1]

    # 1. 右移 target 作为 decoder 输入
    decoder_input = shift_targets_right_with_start_token(tgt_batch, start_id)

    # 2. 前向传播
    log_probs = run_transformer_forward(src_batch, decoder_input, model_params, num_heads, pad_id)

    # 3. 构建平滑标准答案
    batch_size, tgt_seq = tgt_batch.shape
    shape = (batch_size, tgt_seq, vocab_size)
    target = build_uniform_smoothing_distribution(shape, vocab_size, smoothing)
    target = set_confidence_on_gold_tokens(target, tgt_batch, 1.0 - smoothing)
    target = zero_pad_column_and_pad_token_rows(target, tgt_batch, pad_id)

    # 4. 算 loss
    total_loss = compute_label_smoothed_kl_loss(log_probs, target)

    # 5. 平均
    loss = average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)

    return loss

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

