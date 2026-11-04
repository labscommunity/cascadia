# Portions derived from Cloudflare's Clef (https://huggingface.co/Cloudflare/clef-flash),
# `joint_schema_model.py`, which carries no copyright header; see THIRD_PARTY_NOTICES.md.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
#
# `EvidenceRoutingLayer` and `JointSchemaHead` are copied verbatim from Cloudflare's
# Clef `joint_schema_model.py` (Apache-2.0). `TensorHead` is the same math over flat
# tensors (from the clef-spike `step2_head_export.py`), with `cosine_similarity`
# rewritten because OpenVINO has no conversion rule for it. The OpenVINO export
# helpers and loaders below are additions.
"""Clef joint schema head (PyTorch) and its OpenVINO export.

The exported IR's inputs are the contract with crates/cascadia-engine-openvino/src/decision.rs:
  hidden [T,H] f32, span_mask [Q+O,T] f32, lexical [O,H] f32, option_q [O] i64, type_ids [Q] i64
and its single output is [O] f32 (one logit per option; softmax per question outside).
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as functional
import torch.nn.functional as F

# Upstream's joint_schema_model.EncodedRecord (an encode_record result), named by the verbatim
# JointSchemaHead.forward signature below; only its .questions is read.
EncodedRecord = Any


class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        routed, _ = self.attention(
            normalized_queries,
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [
                EvidenceRoutingLayer(
                    width=width,
                    heads=heads,
                    feedforward=feedforward,
                    dropout=dropout,
                )
                for _ in range(routing_layers)
            ]
        )
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1),
        )
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        start, end = span
        return values[start:end].mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list[EncodedRecord],
        output_embedding_weight: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        results: list[list[torch.Tensor]] = []
        normalized_hidden = self.hidden_norm(hidden_states)
        for batch_index, record in enumerate(records):
            sequence_length = int(attention_mask[batch_index].sum().item())
            sequence_hidden = normalized_hidden[batch_index, :sequence_length]
            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]
            question_vectors = torch.stack(
                [
                    self._mean_span(sequence_hidden, question.question_span)
                    for question in record.questions
                ]
            )
            type_ids = torch.tensor(
                [question.question_type for question in record.questions],
                device=hidden_states.device,
            )
            option_contexts: list[torch.Tensor] = []
            lexical_options: list[torch.Tensor] = []
            option_counts = []
            for question in record.questions:
                context_vectors = torch.stack(
                    [
                        self._mean_span(sequence_hidden, span)
                        for span in question.option_spans
                    ]
                )
                lexical_vectors = []
                for start, end in question.option_spans:
                    token_ids = input_ids[batch_index, start:end]
                    lexical_vectors.append(output_embedding_weight[token_ids].mean(dim=0))
                lexical = torch.stack(lexical_vectors)
                option_contexts.append(context_vectors)
                lexical_options.append(lexical)
                option_counts.append(len(question.option_spans))

            option_queries = []
            for question_index, (context_vectors, lexical) in enumerate(
                zip(option_contexts, lexical_options)
            ):
                option_queries.append(
                    self.option_context_projection(context_vectors)
                    + self.option_lexical_projection(lexical)
                    + self.option_question_projection(
                        question_vectors[question_index]
                    ).unsqueeze(0)
                )
            routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed_options = layer(routed_options, memory)
            routed_options = routed_options[0]
            split_options = list(torch.split(routed_options, option_counts, dim=0))

            base_fields = self.question_projection(question_vectors)
            option_summaries = []
            for field, options in zip(base_fields, split_options):
                routing_weights = torch.softmax(
                    torch.matmul(options, field) / math.sqrt(options.shape[-1]),
                    dim=0,
                )
                option_summaries.append(
                    torch.sum(routing_weights.unsqueeze(-1) * options, dim=0)
                )
            fields = (
                base_fields
                + self.option_summary_norm(torch.stack(option_summaries))
                + self.global_projection(global_vector).unsqueeze(0)
                + self.type_embedding(type_ids)
            )
            fields = fields.unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            record_logits: list[torch.Tensor] = []
            for field, question, lexical, routed in zip(
                fields,
                record.questions,
                lexical_options,
                split_options,
            ):
                anchor = functional.normalize(
                    question_vectors[len(record_logits)] + global_vector,
                    dim=-1,
                )
                lexical_anchor = functional.normalize(lexical, dim=-1)
                prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
                prior = prior_scale * torch.matmul(lexical_anchor, anchor)
                options = self.option_norm(routed)
                repeated_field = field.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(repeated_field, options, dim=-1)
                features = torch.cat(
                    [
                        repeated_field,
                        options,
                        repeated_field * options,
                        torch.abs(repeated_field - options),
                    ],
                    dim=-1,
                )
                residual = self.residual_scorer(features).squeeze(-1)
                joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
                joint = joint_scale * cosine + residual
                record_logits.append(
                    prior + torch.sigmoid(self.residual_gate) * joint
                )
            results.append(record_logits)
        return results


class TensorHead(torch.nn.Module):
    """JointSchemaHead.forward for one record, over flat tensors.

    hidden:      [T, H]  backbone last_hidden_state (post final norm)
    span_mask:   [Q+O, T] row i = 1/len over span i (questions first, then options)
    lexical:     [O, H]  mean of lm_head rows over each option's tokens
    option_q:    [O]     question index of each option (int64)
    type_ids:    [Q]     question types (int64)
    returns      [O]     one logit per option; softmax per question outside
    """

    def __init__(self, head: JointSchemaHead):
        super().__init__()
        self.h = head

    def forward(self, hidden, span_mask, lexical, option_q, type_ids):
        h = self.h
        q_count = type_ids.shape[0]
        seq = h.hidden_norm(hidden)
        memory = h.memory_projection(seq).unsqueeze(0)
        global_vector = seq[-1]
        pooled = span_mask @ seq
        question_vectors, option_contexts = pooled[:q_count], pooled[q_count:]

        routed = (
            h.option_context_projection(option_contexts)
            + h.option_lexical_projection(lexical)
            + h.option_question_projection(question_vectors)[option_q]
        ).unsqueeze(0)
        for layer in h.evidence_layers:
            routed = layer(routed, memory)
        routed = routed[0]

        # [Q, O] membership; masked softmax == the reference's per-question softmax.
        member = option_q.unsqueeze(0) == torch.arange(q_count).unsqueeze(1)
        base_fields = h.question_projection(question_vectors)
        scores = (base_fields @ routed.T) / math.sqrt(routed.shape[-1])
        weights = torch.softmax(scores.masked_fill(~member, float("-inf")), dim=1)
        fields = (
            base_fields
            + h.option_summary_norm(weights @ routed)
            + h.global_projection(global_vector).unsqueeze(0)
            + h.type_embedding(type_ids)
        ).unsqueeze(0)
        for layer in h.layers:
            fields = layer(fields, memory)
        fields = h.field_norm(fields[0])

        anchor = F.normalize(question_vectors + global_vector, dim=-1)[option_q]
        prior_scale = h.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        prior = prior_scale * (F.normalize(lexical, dim=-1) * anchor).sum(-1)
        options = h.option_norm(routed)
        field = fields[option_q]
        # F.cosine_similarity has no OpenVINO rule. torch clamps each norm at eps=1e-8; this
        # clamps their product at eps^2 instead, so the two agree unless a norm is below ~1e-8
        # (both vectors come out of LayerNorms: field_norm and option_norm).
        cosine = (field * options).sum(-1) / torch.sqrt(
            ((field * field).sum(-1) * (options * options).sum(-1)).clamp_min(1e-16)
        )
        features = torch.cat([field, options, field * options, (field - options).abs()], dim=-1)
        residual = h.residual_scorer(features).squeeze(-1)
        joint_scale = h.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        return prior + torch.sigmoid(h.residual_gate) * (joint_scale * cosine + residual)


def reference_logits(head, hidden, ids, q_spans, o_spans, qtypes, lm_head):
    """Run the reference loop head on one record described by raw spans (test helper)."""
    from types import SimpleNamespace
    questions = [SimpleNamespace(question_span=q, option_spans=tuple(o), question_type=t)
                 for q, o, t in zip(q_spans, o_spans, qtypes)]
    rec = SimpleNamespace(questions=tuple(questions))
    with torch.no_grad():
        out = head(hidden.unsqueeze(0), ids.unsqueeze(0), torch.ones(1, hidden.shape[0], dtype=torch.long), [rec], lm_head)[0]
    return torch.cat(out)


def export_module(module, hidden_size, out_xml):
    import openvino as ov
    t, q, o = 12, 2, 4
    example = (torch.randn(t, hidden_size), torch.full((q + o, t), 1.0 / t), torch.randn(o, hidden_size),
               torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0]))
    traced = torch.jit.trace(module.eval(), example, check_trace=False)
    model = ov.convert_model(traced, example_input=example,
                             input=[ov.PartialShape([-1, hidden_size]), ov.PartialShape([-1, -1]),
                                    ov.PartialShape([-1, hidden_size]), ov.PartialShape([-1]), ov.PartialShape([-1])])
    for inp, name in zip(model.inputs, ["hidden", "span_mask", "lexical", "option_q", "type_ids"]):
        inp.get_tensor().set_names({name})
    ov.save_model(model, out_xml, compress_to_fp16=False)


def load_joint_head(head_dir):
    """head_dir holds Cloudflare's joint_head.safetensors + joint_head_config.json."""
    import json
    from safetensors.torch import load_file
    with open(os.path.join(head_dir, "joint_head_config.json")) as f:
        cfg = json.load(f)
    head = JointSchemaHead(**cfg)
    head.load_state_dict(load_file(os.path.join(head_dir, "joint_head.safetensors")), strict=True)
    return head.float().eval()


def export_joint_head(head_dir, out_xml):
    head = load_joint_head(head_dir)
    export_module(TensorHead(head), head.hidden_norm.normalized_shape[0], out_xml)
