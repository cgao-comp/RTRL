import os, random, copy
from typing import List

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from model.llm_prompt import task_info, wikipedia_info, reddit_info, mooc_info, bitcoin_info, amazon_info
from model.cross_attention import CrossAttention
from model.Qwen_simpled2 import Qwen3AttentionCore
from paths import CKPT_DIR


def ClipLLM_Qwen(original_layers, k_attention=3):
    layers = []

    for idx, layer in enumerate(original_layers[:k_attention]):
        layers.append(layer)

    return nn.Sequential(*layers)


class QwenModifiedModel(nn.Module):
    def __init__(self, llm_model, llm_dim, out_dim, k_attention=3):
        super().__init__()

        self.attention_layers = ClipLLM_Qwen(llm_model.model.layers, k_attention)

        self.out_linear = nn.Linear(llm_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.attention_layers(x)

        if isinstance(x, tuple):
            x = x[0]

        x = self.out_linear(x)

        return x


class LLMModel(nn.Module):
    def __init__(self, llm_model_name: str = "Qwen3.5-0_8B",
                 llm_dim: int = 1024, input_dim: int = 256, emb_dim: int = 1024, out_dim: int = 256, n_heads: int = 8,
                 device: str or torch.device = "cuda:0", dataset_name: str = "wikipedia",
                 config=None):
        super(LLMModel, self).__init__()
        self.device = device
        self.n_heads = n_heads
        self.dataset_name = dataset_name.lower()
        self.config = config
        self.keep_layers = config.llm_keep_layers

        self.task_prompt = task_info
        self.dataset_prompt = "This is a CTDG Dataset."
        if "wikipedia" in self.dataset_name:
            self.dataset_prompt = wikipedia_info
        elif "reddit" in self.dataset_name:
            self.dataset_prompt = reddit_info
        elif "mooc" in self.dataset_name:
            self.dataset_prompt = mooc_info
        elif "bitcoin" in self.dataset_name:
            self.dataset_prompt = bitcoin_info
        elif "amazon" in self.dataset_name:
            self.dataset_prompt = amazon_info

        self.dataset_prompt_list = self.dataset_prompt.split(' ')
        self.dataset_prompt_length = len(self.dataset_prompt_list)
        self.task_prompt_length = len(self.task_prompt)
        self.mask_length = self.dataset_prompt_length + 1

        self.llm_arch = config.llm_arch
        self.llm_data_type = torch.bfloat16 if config.llm_dtype == 'bfloat16' else torch.float32
        self.out_data_type = torch.float32
        llm_model = None

        if llm_model_name == 'Qwen3-0_6B':

            model_name = 'Qwen/Qwen3-0.6B'

            local_model_dir = os.path.join(CKPT_DIR, 'LLM', 'Qwen3-0_6B')
            self.llm_config = AutoConfig.from_pretrained(os.path.join(local_model_dir, "config.json"))

            try:
                model_files_exist = os.path.exists(os.path.join(local_model_dir, 'pytorch_model.bin')) or \
                                    os.path.exists(os.path.join(local_model_dir, 'model.safetensors'))

                if model_files_exist:
                    print(f"Loading transformers model from local directory: {local_model_dir}")
                    llm_model = AutoModelForCausalLM.from_pretrained(
                        local_model_dir,
                        trust_remote_code=True,
                        config=self.llm_config,
                        attn_implementation="eager"
                    )
                else:
                    print(f"Local model files not found at {local_model_dir}. Downloading from HuggingFace...")
                    llm_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        config=self.llm_config,
                        attn_implementation="eager"
                    )
                    llm_model.save_pretrained(local_model_dir)
                    print(f"Model saved to {local_model_dir}")
            except Exception as e:
                print(f"Error loading model: {e}")
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    config=self.llm_config,
                    attn_implementation="eager"
                )

            try:
                tokenizer_files_exist = os.path.exists(os.path.join(local_model_dir, 'vocab.txt')) or \
                                        os.path.exists(os.path.join(local_model_dir, 'tokenizer_config.json'))

                if tokenizer_files_exist:
                    print(f"Loading tokenizer from local directory: {local_model_dir}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        local_model_dir,
                        trust_remote_code=True,
                    )
                else:
                    print(f"Local tokenizer files not found at {local_model_dir}. Downloading from HuggingFace...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                    )
                    self.tokenizer.save_pretrained(local_model_dir)
                    print(f"Tokenizer saved to {local_model_dir}")
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
        else:
            model_name = llm_model_name
            local_model_dir = os.path.join(CKPT_DIR, 'LLM', llm_model_name)
            os.makedirs(local_model_dir, exist_ok=True)
            self.llm_config = AutoConfig.from_pretrained(os.path.join(local_model_dir, "config.json"))

            try:
                model_files_exist = os.path.exists(os.path.join(local_model_dir, 'pytorch_model.bin')) or \
                                    os.path.exists(os.path.join(local_model_dir, 'model.safetensors')) or \
                                    os.path.exists(os.path.join(local_model_dir, 'model.safetensors.index.json'))

                if model_files_exist:
                    print(f"Loading transformers model from local directory: {local_model_dir}")
                    llm_model = AutoModelForCausalLM.from_pretrained(
                        local_model_dir,
                        trust_remote_code=True,
                        config=self.llm_config,
                        attn_implementation="eager"
                    )
                else:
                    print(f"Local model files not found at {local_model_dir}. Downloading from HuggingFace...")
                    llm_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        config=self.llm_config,
                        attn_implementation="eager"
                    )
                    llm_model.save_pretrained(local_model_dir)
                    print(f"Model saved to {local_model_dir}")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise Exception('LLM model is not defined')

            try:
                tokenizer_files_exist = os.path.exists(os.path.join(local_model_dir, 'vocab.txt')) or \
                                        os.path.exists(os.path.join(local_model_dir, 'tokenizer_config.json'))

                if tokenizer_files_exist:
                    print(f"Loading tokenizer from local directory: {local_model_dir}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        local_model_dir,
                        trust_remote_code=True,
                    )
                else:
                    print(f"Local tokenizer files not found at {local_model_dir}. Downloading from HuggingFace...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                    )
                    self.tokenizer.save_pretrained(local_model_dir)
                    print(f"Tokenizer saved to {local_model_dir}")
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )

        if self.llm_arch == 'qwen':
            for param in llm_model.parameters():
                param.requires_grad = False

            self.llm_input_embeddings = llm_model.get_input_embeddings().to(self.llm_data_type).to(self.device)

            self.llm_out_fc = nn.ModuleList()
            for i in range(self.keep_layers):
                out_tp = Qwen3AttentionCore(
                    hidden_size=1024,
                    num_heads=16,
                    num_kv_heads=8,
                    head_dim=128,
                )
                if not self.config.randn_qwen:
                    out_tp.copy_weight(llm_model.model.layers[i])
                for param in out_tp.parameters():
                    param.requires_grad = False
                self.llm_out_fc.append(out_tp.to(self.llm_data_type).to(self.device))
            self.out_linear2 = nn.Linear(llm_dim, out_dim, bias=False, dtype=self.out_data_type).to(self.device)
        else:
            self.llm_input_embeddings = llm_model.get_input_embeddings().to(self.llm_data_type)
            if self.config.llm_freeze_emb:
                for param in self.llm_input_embeddings.parameters():
                    param.requires_grad = False

        del llm_model
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        self.word_embeddings = self.llm_input_embeddings.weight
        self.vocab_size, self.vocab_dim = self.word_embeddings.shape
        self.num_tokens = self.config.num_tokens
        self.trans_layer = nn.Linear(self.vocab_size, self.num_tokens, bias=False,
                                     dtype=self.llm_data_type).to(self.device)
        self.word_linear = nn.Linear(self.vocab_dim, emb_dim, bias=False,
                                     dtype=self.llm_data_type).to(self.device)

        self.x_linear = nn.Linear(input_dim, emb_dim, bias=False).to(self.device)

        if self.llm_arch == 'mlp':
            self.out_linear = nn.Sequential(
                nn.Linear(llm_dim, out_dim, bias=False, dtype=self.out_data_type),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim, bias=False, dtype=self.out_data_type),
            )

        self.cross_attn = CrossAttention(
            input_dim=emb_dim,
            hidden_dim=emb_dim,
            out_dim=llm_dim,
            n_heads=self.n_heads,
            dropout=0.1,
            data_type=self.llm_data_type
        )

    def forward(self, x,
                source_nodes: List[int] or torch.Tensor,
                dest_nodes: List[int] or torch.Tensor,
                time_list: List[int] or torch.Tensor):
        batch_size, x_seq_size, _ = x.shape

        assert len(source_nodes) == len(dest_nodes)
        assert len(source_nodes) == len(time_list)

        prompt_str_batch = []
        for i in range(batch_size):
            src_node = source_nodes[i]
            dst_node = dest_nodes[i]
            event_time = time_list[i]

            prompt_str = generate_prompt(self.task_prompt, self.dataset_prompt,
                                         event_time, src_node, dst_node)
            prompt_str_batch.append(prompt_str)

        prompt_tokens = self.tokenizer(prompt_str_batch, return_tensors="pt", padding=True, truncation=True,
                                       max_length=2048).input_ids
        prompt_embs = self.llm_input_embeddings(prompt_tokens.to(self.device))

        source_embeddings = self.trans_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        enc_out = self.x_linear(x).to(self.llm_data_type)
        source_embeddings = self.word_linear(source_embeddings)
        enc_out = self.cross_attn(enc_out, source_embeddings, source_embeddings)

        llama_enc_out = torch.cat([prompt_embs, enc_out], dim=1)
        dec_out = self.head(llama_enc_out)

        return (dec_out[:, -x_seq_size:, :])

    @torch.no_grad()
    def infer(self, x,
              source_nodes: List[int] or torch.Tensor,
              dest_nodes: List[int] or torch.Tensor,
              time_list: List[int] or torch.Tensor):
        batch_size, x_seq_size, _ = x.shape

        assert len(source_nodes) == len(dest_nodes)
        assert len(source_nodes) == len(time_list)

        prompt_str_batch = []
        for i in range(batch_size):
            src_node = source_nodes[i]
            dst_node = dest_nodes[i]
            event_time = time_list[i]

            prompt_str = generate_prompt(self.task_prompt, self.dataset_prompt_list,
                                         event_time, src_node, dst_node)
            prompt_str_batch.append(prompt_str)

        prompt_tokens = self.tokenizer(prompt_str_batch, return_tensors="pt", padding=True, truncation=True,
                                       max_length=2048).input_ids
        prompt_embs = self.llm_input_embeddings(prompt_tokens.to(self.device))
        source_embeddings = self.trans_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        enc_out = self.x_linear(x).to(self.llm_data_type)
        source_embeddings = self.word_linear(source_embeddings)
        enc_out = self.cross_attn(enc_out, source_embeddings, source_embeddings)

        llama_enc_out = torch.cat([prompt_embs, enc_out], dim=1)
        dec_out = self.head(llama_enc_out)

        return dec_out[:, -x_seq_size:, :]

    def head(self, x):
        if self.llm_arch == 'qwen':
            with torch.no_grad():
                x = self.llm_forward(x)
            return self.out_linear2(x.to(self.out_data_type))
        return self.out_linear(x.to(self.out_data_type))

    @torch.no_grad()
    def llm_forward(self, x):
        assert self.llm_out_fc is not None
        for i in range(self.keep_layers):
            x = self.llm_out_fc[i](x)
        return x


def generate_prompt(task_prompt, dataset_prompt, event_time, src_node, dst_node):
    return (f"<|start_prompt|>Task description: {task_prompt}; "
            f"Data information: {dataset_prompt}.; "
            f"Input information: "
            f"Timestamp {event_time}, "
            f"Source Node {src_node} is connected to Node {dst_node}<|<end_prompt>|>")
