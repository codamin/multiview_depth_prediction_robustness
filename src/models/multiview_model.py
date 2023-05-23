from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers import DPTForDepthEstimation, DPTModel
from transformers.models.dpt.modeling_dpt import DPTViTLayer, DPTConfig

class Positional3DEncoder(nn.Module):
    def __init__(self, depth_size, patch_size, hidden_size):
        super().__init__()
        self.conv = torch.nn.Conv2d(depth_size * 3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.mlp_layer_1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.mlp_layer_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.zero_init()
    
    def forward(self, pos):
        pos = self.conv(pos)
        pos = torch.cos(pos)
        pos = self.mlp_layer_1(pos)
        pos = torch.selu(pos)
        pos = self.mlp_layer_2(pos)
        pos = pos.flatten(2).transpose(1, 2)
        return pos
    
    def zero_init(self):
        nn.init.zeros_(self.mlp_layer_1.weight)
        nn.init.zeros_(self.mlp_layer_1.bias)
        nn.init.zeros_(self.mlp_layer_2.weight)
        nn.init.zeros_(self.mlp_layer_2.bias)


class DPTMultiviewViTEncoder(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([DPTViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        knowledge_sources: Tuple[torch.Tensor], # knowledge_sources is tuple with size layer_num. each element of the tuple has the shape batch_size, seq_length, hidden_size
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_knowledge_sources: bool = False,
    ) -> dict:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_knowledge_sources = () if output_knowledge_sources else None

        hidden_states_length = hidden_states.shape[1]

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            hidden_states = torch.cat((hidden_states, knowledge_sources[i]), dim=1)
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            layer_output = layer_outputs[0]

            hidden_states, ks_output = layer_output[: , :hidden_states_length], layer_output[: , hidden_states_length:] 

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if output_knowledge_sources:
                all_knowledge_sources = all_knowledge_sources + (ks_output,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        output = {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "knowledge_sources": knowledge_sources,
        }
        
        return output


class DPTMultiviewModel(DPTModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.encoder = DPTMultiviewViTEncoder(config)

        if config.pos3d_encoding:
            self.pos3d_encoder = Positional3DEncoder(depth_size=config.pos3d_depth, patch_size=config.patch_size, hidden_size=config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        knowledge_sources: Tuple[torch.Tensor],
        points3d: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> dict:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, return_dict=True)

        embedding_last_hidden_states = embedding_output.last_hidden_states

        if hasattr(self, "pos3d_encoder") and not torch.isnan(points3d).any():
            pos3d_encoding = self.pos3d_encoder(points3d)
            print(pos3d_encoding.sum())
            embedding_last_hidden_states[:, :-1] = embedding_last_hidden_states[:, :-1] + pos3d_encoding

        encoder_outputs = self.encoder(
            embedding_last_hidden_states,
            knowledge_sources,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_knowledge_sources=True,
        )

        sequence_output = encoder_outputs["last_hidden_state"]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None


        output = {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
            "intermediate_activations": embedding_output.intermediate_activations,
            "knowledge_sources": encoder_outputs["knowledge_sources"],
        }

        return output


class DPTMultiviewDepth(DPTForDepthEstimation):
    def __init__(self, config, num_seq_knowledge_source=200, pos3d_encoding=True, pos3d_depth=5):
        super().__init__(config)

        if not hasattr(config, "num_seq_knowledge_source"):
            config.__setattr__("num_seq_knowledge_source", num_seq_knowledge_source)
        if not hasattr(config, "pos3d_encoding"):
            config.__setattr__("pos3d_encoding", pos3d_encoding)
        if not hasattr(config, "pos3d_depth"):
            config.__setattr__("pos3d_depth", pos3d_depth)

        del self.dpt
        self.dpt = DPTMultiviewModel(config, add_pooling_layer=False)

        self.knowledge_sources = nn.ParameterList([torch.rand(1, config.num_seq_knowledge_source, config.hidden_size) for _ in range(config.num_hidden_layers)])

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        pixel_values: torch.FloatTensor,
        knowledge_sources: Optional[Tuple[torch.Tensor]] = None,
        points3d: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> dict:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if knowledge_sources is None:
            knowledge_sources = self.init_knowledge_source(batch_size=pixel_values.shape[0])

        dpt_outputs = self.dpt(
            pixel_values,
            knowledge_sources,
            points3d,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
        )

        hidden_states = dpt_outputs["hidden_states"]

        # only keep certain features based on config.backbone_out_indices
        # note that the hidden_states also include the initial embeddings
        if not self.config.is_hybrid:
            hidden_states = [
                feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices
            ]
        else:
            backbone_hidden_states = dpt_outputs["intermediate_activations"]
            backbone_hidden_states.extend(
                feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices[2:]
            )

            hidden_states = backbone_hidden_states

        hidden_states = self.neck(hidden_states)

        predicted_depth = self.head(hidden_states)

        output = {
            "predicted_depth": predicted_depth,
            "hidden_states": dpt_outputs["hidden_states"] if output_hidden_states else None,
            "attentions": dpt_outputs["attentions"],
            "knowledge_sources": dpt_outputs["knowledge_sources"],
        }
        
        return output
            
    
    def init_knowledge_source(self, batch_size):
        ks_tuple = ()
        for ks in self.knowledge_sources:
            ks_tuple = ks_tuple + (ks.expand(batch_size, -1, -1),)
        return ks_tuple
    

if __name__=='__main__':
    model = DPTMultiviewDepth.from_pretrained("Intel/dpt-large")
    out = model(torch.zeros((1,3,384,384)), points3d=torch.zeros((1,15,384,384)))